import torch
from rendiff.utils.dataset import RGBDataset
from torch.utils.data import DataLoader

from rendiff.models.diffusion import Diffusion
from diffusers.optimization import get_scheduler
import os
import lightning as L
import argparse
from rendiff.models.backbone import Backbone
from rendiff.configs.rendif_config import CONFIG as config
import copy
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor
from rendiff.utils.common_utils import printarr
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random



# Lightening wrapper for the model.
class LitModel(L.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.save_weights_only = True

        self.loss_f = torch.nn.L1Loss()
        self.loss_grip = torch.nn.BCEWithLogitsLoss(pos_weight=config['action_stats']['pos_weight'])

        self.loss_g_history = [] 
        self.loss_a_history = []
        self.loss_p_history = []
        self.loss_history = []
        self.camera_loss_history = {
            camera_name:[] for camera_name in config['camera_names']
        }
        self.loss_g_history_val = [] 
        self.loss_a_history_val = []
        self.loss_p_history_val = []
        self.loss_history_val = []
        self.camera_loss_history_val = {
            camera_name:[] for camera_name in config['camera_names']
        }
        self.writer = SummaryWriter()
    
    def training_step(self, data):
        model_ret = self.model(data, data['action'])

        labels, masks, preds, points, grips, labels_grip = (model_ret['labels'],
                                                            model_ret['masks'],
                                                            model_ret['preds'],
                                                            model_ret['points'],
                                                            model_ret['pred_g'],
                                                            model_ret['labels_g'])

        loss_g = self.loss_grip(grips * data['is_valid'],
                                labels_grip * data['is_valid'])

        self.log("Loss_g", loss_g, on_step=True, on_epoch=True, prog_bar=True)
        self.loss_g_history.append(loss_g.item()) 

        if self.config['pred_actions']:
            a_loss = self.loss_f(model_ret['pred_actions'] * data['is_valid'],
                                 model_ret['labels_actions'] * data['is_valid'])
            self.log("Loss_a", a_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.loss_a_history.append(a_loss.item()) 

        pixel_loss = 0
        for key in preds.keys():
            camera_loss = self.loss_f((preds[key] * data['is_valid'].unsqueeze(-1).unsqueeze(-1))[masks[key]],
                                      (labels[key] * data['is_valid'].unsqueeze(-1).unsqueeze(-1))[masks[key]])

            if self.config['mask_pixel_weight'] > 0:
                camera_loss += self.config['mask_pixel_weight'] * self.loss_f(preds[key][~masks[key]],
                                                                              labels[key][~masks[key]])
            if not torch.isnan(camera_loss):  # Can happen if all pixels are masked.
                pixel_loss += camera_loss
            
            self.log(f"Loss_Pixel/{key}", camera_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.camera_loss_history[key].append(camera_loss.item())            
            
        pixel_loss = pixel_loss/3
        self.log("Loss_p", pixel_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.loss_p_history.append(pixel_loss.item())  

        loss = pixel_loss + self.config['gripper_loss_weight'] * loss_g
        if self.config['pred_actions']:
            loss += self.config['action_loss_weight'] * a_loss
        self.loss_history.append(loss.item()) 

        return loss

    def validation_step(self, data):
        self.model.net.proc.renderer.extended_meshes = None
        self.model.net.proc.renderer_wrist.extended_meshes = None
        # model_ret, noisy_action = model(data, None)
        model_ret = self.model(data, data['action'])
        
        labels, masks, preds, points, grips, labels_grip = (model_ret['labels'],
                                                            model_ret['masks'],
                                                            model_ret['preds'],
                                                            model_ret['points'],
                                                            model_ret['pred_g'],
                                                            model_ret['labels_g'])

        loss_g_val = self.loss_grip(grips * data['is_valid'],
                                labels_grip * data['is_valid'])

        self.log("Loss_g_val", loss_g_val, on_step=True, on_epoch=True, prog_bar=True)
        self.loss_g_history_val.append(loss_g_val.item()) 

        if self.config['pred_actions']:
            a_loss_val = self.loss_f(model_ret['pred_actions'] * data['is_valid'],
                                 model_ret['labels_actions'] * data['is_valid'])
            self.log("Loss_a_val", a_loss_val, on_step=True, on_epoch=True, prog_bar=True)
            self.loss_a_history_val.append(a_loss_val.item()) 

        pixel_loss_val = 0
        for key in preds.keys(): 
            camera_loss_val = self.loss_f((preds[key] * data['is_valid'].unsqueeze(-1).unsqueeze(-1))[masks[key]],
                                      (labels[key] * data['is_valid'].unsqueeze(-1).unsqueeze(-1))[masks[key]])

            if self.config['mask_pixel_weight'] > 0:
                camera_loss_val += self.config['mask_pixel_weight'] * self.loss_f(preds[key][~masks[key]],
                                                                              labels[key][~masks[key]])
            if not torch.isnan(camera_loss_val):  # Can happen if all pixels are masked.
                pixel_loss_val += camera_loss_val
            
            self.log(f"Loss_Pixel_val/{key}", camera_loss_val, on_step=True, on_epoch=True, prog_bar=True)
            self.camera_loss_history_val[key].append(camera_loss_val.item())            
            
        if pixel_loss_val!=0:
            pixel_loss_val = pixel_loss_val/3
            self.log("Loss_p_val", pixel_loss_val, on_step=True, on_epoch=True, prog_bar=True)
            self.loss_p_history_val.append(pixel_loss_val.item())  

        loss_val = pixel_loss_val + self.config['gripper_loss_weight'] * loss_g_val
        if self.config['pred_actions']:
            loss_val += self.config['action_loss_weight'] * a_loss_val
        self.loss_history_val.append(loss_val.item()) 
        
        self.model.net.proc.renderer.extended_meshes = None
        self.model.net.proc.renderer_wrist.extended_meshes = None

    def on_train_epoch_end(self):
        # Manually save the model at the end of each epoch.
        if self.config['record'] and self.current_epoch % self.config['save_every_epochs'] == 0:
            self.save_current_model(f'{save_root}/runs/{run_name}/{run_name}_final.pt')
            
        # Log the loss to TensorBoard (based on epoch)
        self.writer.add_scalar("Epoch_Train/Gripper", np.mean(self.loss_g_history), self.global_step)
        self.writer.add_scalar("Epoch_Train/Action", np.mean(self.loss_a_history), self.global_step)
        self.writer.add_scalar("Epoch_Train/Pixel", np.mean(self.loss_p_history), self.global_step)       
        self.writer.add_scalar("Epoch_Train/Total", np.mean(self.loss_history), self.global_step)       
        for key in self.camera_loss_history.keys():
            self.writer.add_scalar(f"Epoch_Train/Pixel_{key}", np.mean(self.camera_loss_history[key]), self.global_step)
        self.writer.add_scalar(f"Epoch_Train/Pixel_Average", np.mean(self.loss_p_history)/len(self.camera_loss_history.keys()), self.global_step)
        
        self.loss_g_history = [] 
        self.loss_a_history = []
        self.loss_p_history = []
        self.loss_history = []
        self.camera_loss_history = {
            camera_name:[] for camera_name in config['camera_names']
        }
        
    def on_validation_epoch_end(self):
        self.writer.add_scalar("Epoch_Val/Gripper", np.mean(self.loss_g_history_val), self.global_step)
        self.writer.add_scalar("Epoch_Val/Action", np.mean(self.loss_a_history_val), self.global_step)
        self.writer.add_scalar("Epoch_Val/Pixel", np.mean(self.loss_p_history_val), self.global_step)       
        self.writer.add_scalar("Epoch_Val/Total", np.mean(self.loss_history_val), self.global_step)       
        for key in self.camera_loss_history_val.keys():
            self.writer.add_scalar(f"Epoch_Val/Pixel_{key}", np.mean(self.camera_loss_history_val[key]), self.global_step)
        self.writer.add_scalar(f"Epoch_Val/Pixel_Average", np.mean(self.loss_p_history_val)/len(self.camera_loss_history_val.keys()), self.global_step)
        
        self.loss_g_history_val = [] 
        self.loss_a_history_val = []
        self.loss_p_history_val = []
        self.loss_history_val = []
        self.camera_loss_history_val = {
            camera_name:[] for camera_name in config['camera_names']
        }

    def save_current_model(self, save_path):
        checkpoint = {
            'state_dict': copy.deepcopy(self.model.state_dict()),
        }

        if not self.save_weights_only:
            checkpoint['lightning_checkpoint'] = self.trainer._checkpoint_connector.dump_checkpoint(False)

        # A hack to save the model with the EMA weights.
        if self.config['use_ema']:
            self.model.ema.copy_to(self.model.parameters())
            checkpoint['state_dict_ema'] = self.model.state_dict()
            # Reload original weights, not EMA weights
            self.model.load_state_dict(checkpoint['state_dict'])
            if not self.save_weights_only:
                checkpoint['ema_state_dict'] = self.model.ema.state_dict()
        torch.save(checkpoint, save_path)

    def on_before_zero_grad(self, *args, **kwargs):
        if self.config['use_ema']:
            self.model.ema.step(model.parameters())

    def on_train_batch_end(self, *args, **kwargs):
        if self.global_step in self.config['save_itt'] and self.config['record']:
            self.save_current_model(f'{save_root}/runs/{run_name}/{run_name}_{self.global_step}_step.pt')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config['lr'],
                                      weight_decay=self.config['weight_decay'])
        if not self.config['use_scheduler']:
            return optimizer

        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=self.config['num_iters'],
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step", "frequency": 1}]


if __name__ == '__main__':
    ####################################################################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', type=int, default=1)
    parser.add_argument('--use_wandb', type=int, default=0)
    parser.add_argument('--reprocess', type=int, default=1)
    parser.add_argument('--datadir', type=str, default='./data/phone_on_base')
    parser.add_argument('--num_demos', type=int, default=20)
    parser.add_argument('--processed_dir', type=str, default='processed')
    parser.add_argument('--run_name', type=str, default='TEST')
    parser.add_argument('--save_root', type=str, default='.')
    
    seed = 42  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed) 
    print(f'Setting random seed to {seed}')

    record = bool(parser.parse_args().record)
    use_wandb = bool(parser.parse_args().use_wandb)
    data_path = parser.parse_args().datadir
    num_demos = parser.parse_args().num_demos
    processed_dir = parser.parse_args().processed_dir
    run_name = parser.parse_args().run_name
    save_root = parser.parse_args().save_root
    reprocess = bool(parser.parse_args().reprocess)
    ####################################################################################################################
    config['record'] = record
    config['save_root'] = save_root
    ####################################################################################################################
    dataset = RGBDataset(root=data_path, reprocess=reprocess, camera_names=config['camera_names'],
                         pred_horizon=config['pred_horizon'], num_demos=num_demos, filter_demos=config['filter_demos'],
                         action_history=config['action_history'], processed_dir=processed_dir,
                         oversample_changes=config['oversample_changes'],
                         randomize_g_prob=config['randomize_g_prob'])
    print(f'Number of samples: {len(dataset)}')
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=8,
                            pin_memory=True)
    ####################################################################################################################
    # Add a validation set here. For now, we just use the same dataset to monitor loss on predictions after diffusing.
    val_path = os.path.join(data_path, "val")
    dataset_val = RGBDataset(root=val_path, reprocess=reprocess, camera_names=config['camera_names'],
                             pred_horizon=config['pred_horizon'], num_demos=15,
                             filter_demos=config['filter_demos'],
                             action_history=config['action_history'], processed_dir=processed_dir+'_val',
                             oversample_changes=config['oversample_changes'],)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, drop_last=False, num_workers=1)
    ####################################################################################################################
    action_stats = torch.load(f'{data_path}/{processed_dir}/action_stats.pt')
    config['action_stats'] = action_stats
    config['intrinsics'] = torch.load(f'{data_path}/{processed_dir}/intrinsics.pt')

    if record:
        os.makedirs(f'{save_root}/runs/{run_name}', exist_ok=True)
        config['save_root'] = save_root
        torch.save(action_stats, f'{save_root}/runs/{run_name}/action_stats.pt')
        torch.save(config, f'{save_root}/runs/{run_name}/config.pt')
        if use_wandb:
            logger = WandbLogger(project='R&D',
                                 name=run_name,
                                 save_dir=f'{save_root}/runs/{run_name}',
                                 log_model=False)
            # We save models manually, so don't need to log them.
            callbacks = [
                LearningRateMonitor(logging_interval='step')
            ]
    ####################################################################################################################
    backbone = Backbone(config)
    model = Diffusion(backbone,
                      num_diffusion_iters=config['num_diffusion_iters_train'],
                      device=config['device'],
                      pred_horizon=config['pred_horizon'],
                      action_dim=7,
                      inference_num_diffusion_iters=3,
                      action_stats=action_stats,
                      config=config,
                      ).to(config['device'])
    ####################################################################################################################
    lit_model = LitModel(model, config)
    trainer = L.Trainer(
        default_root_dir=save_root,
        enable_checkpointing=False,  
        accelerator='gpu',
        devices=1,
        max_steps=config['num_iters'],
        enable_progress_bar=True,
        precision='16-mixed',
        logger=logger if record and use_wandb else None,
        log_every_n_steps=1000,
        val_check_interval=2000,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None,
        callbacks=callbacks if record and use_wandb else None,
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=dataloader,
        val_dataloaders=dataloader_val,
    ) 
    if config['record']:
        lit_model.save_current_model(f'{save_root}/runs/{run_name}/{run_name}_final.pt')
    ####################################################################################################################
