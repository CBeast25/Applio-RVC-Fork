import torch
from torch.cuda.amp import GradScaler, autocast

from lib.infer.infer_libs.train import utils

from lib.infer.infer_libs.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch

from lib.infer.infer_libs.infer_pack import commons

from lib.infer.infer_libs.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from lib.infer.infer_libs.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)
import lib.infer.infer_libs.train.utils as utils
hps = utils.get_hparams()
n_gpus = len(hps.gpus.split("-"))

if hps.version == "v1":
    from lib.infer.infer_libs.infer_pack.models import MultiPeriodDiscriminator
    from lib.infer.infer_libs.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from lib.infer.infer_libs.infer_pack.models import (
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
    )
else:
    from lib.infer.infer_libs.infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

import os
import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import LearningRateFinder
from lightning.pytorch.callbacks import BatchSizeFinder
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class FineTuneBatchSizeFinder(BatchSizeFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.scale_batch_size(trainer, pl_module)

class FineTuneLearningRateFinder(LearningRateFinder):
    def __init__(self, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)

class CustomDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()
        
    def prepare_data(self):
        if hps.if_f0 == 1:
            self.train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
        else:
            self.train_dataset = TextAudioLoader(hps.data.training_files, hps.data)

    def setup(self):
        self.train_sampler = DistributedBucketSampler(
            self.train_dataset,
            hps.train.batch_size * n_gpus,
            # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
            [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
            num_replicas=n_gpus,
            shuffle=True,
        )
        if hps.if_f0 == 1:
            self.collate_fn = TextAudioCollateMultiNSFsid()
        else:
            self.collate_fn = TextAudioCollate()


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            num_workers=os.cpu_count()-1,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
            batch_sampler=self.train_sampler,
            persistent_workers=True,
            prefetch_factor=8,
            batch_size=hps.train.batch_size
        )

class GAN(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        if hps.if_f0 == 1:
            self.generator = RVC_Model_f0(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model,
                is_half=hps.train.fp16_run,
                sr=hps.sample_rate,
            )
        else:
            self.generator = RVC_Model_nof0(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model,
                is_half=hps.train.fp16_run,
            )
        self.discriminator = MultiPeriodDiscriminator(hps.model.use_spectral_norm)

        self.scaler = GradScaler(enabled=hps.train.fp16_run)

    def forward(self, phone, phone_lengths, y, y_lengths, ds):
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, None, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def training_step(self, info, batch_idx):
        if hps.if_f0 == 1:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info

        optim_g, optim_d = self.optimizers()
        scaler = self.scaler
        net_g = self.generator
        net_d = self.discriminator
        # Calculate
        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0 == 1:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run == True:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        lr = optim_g.param_groups[0]["lr"]
        self.log(self.current_epoch, prog_bar=True)
        # Amor For Tensorboard display
        if loss_mel > 75:
            loss_mel = 75
        if loss_kl > 9:
            loss_kl = 9

        self.logger.info([self.global_step, lr])
        self.logger.info(
            f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
        )
        if self.global_step % hps.train.log_interval == 0:
            scalar_dict = {
                "loss/g/total": loss_gen_all,
                "loss/d/total": loss_disc,
                "learning_rate": lr,
                "grad_norm_d": grad_norm_d,
                "grad_norm_g": grad_norm_g,
            }
            scalar_dict.update(
                {
                    "loss/g/fm": loss_fm,
                    "loss/g/mel": loss_mel,
                    "loss/g/kl": loss_kl,
                }
            )

            scalar_dict.update(
                {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
            )
            scalar_dict.update(
                {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
            )
            scalar_dict.update(
                {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
            )
            image_dict = {
                "slice/mel_org": utils.plot_spectrogram_to_numpy(
                    y_mel[0].data.cpu().numpy()
                ),
                "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                    y_hat_mel[0].data.cpu().numpy()
                ),
                "all/mel": utils.plot_spectrogram_to_numpy(
                    mel[0].data.cpu().numpy()
                ),
            }
            for k, v in scalar_dict.items():
                self.logger.experiment.add_scalar(k, v, self.global_step)
            for k, v in image_dict.items():
                self.logger.experiment.add_image(k, v, self.global_step, dataformats="HWC")
        

    def configure_optimizers(self):
        optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
        optim_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=hps.train.lr_decay
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=hps.train.lr_decay
        )
        return [optim_g, optim_d], [scheduler_g, scheduler_d]
    
if __name__ == "__main__":
    logger = TensorBoardLogger("train.log", hps.model_dir)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="loss/g/total",
        mode="min",
        dirpath=hps.model_dir,
        filename="sample-model-{epoch:02d}-{loss_gen_all:.2f}",
    )
    dm = CustomDataModule()
    model = GAN()
    trainer = L.Trainer(
        default_root_dir=hps.model_dir,
        callbacks=[checkpoint_callback,FineTuneBatchSizeFinder(milestones=(5, 10)),FineTuneLearningRateFinder(milestones=(5, 10))],
        accelerator="auto",
        max_epochs=hps.train.epochs + 1,
        logger=logger
    )
    trainer.fit(model, dm)