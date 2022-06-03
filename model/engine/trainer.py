import time
import os
from tqdm import tqdm
import datetime

import torch

from model.utils.postprocess import _transpose_and_gather_feat

from torch import linalg as LA


def do_train(args, cfg, model, optimizer, scheduler, d_optimizer, d_scheduler, scaler, data_loader, summary_writer):
    max_iter = len(data_loader['train']) + args.resume_iter
    trained_time = 0
    tic = time.time()
    end = time.time()

    logging_losses = {}

    detector_params = list(filter(lambda p:p.requires_grad, list(model.module.extractors.parameters()) + list(model.module.detector.parameters())))
    discriminator_params = list(filter(lambda p:p.requires_grad, list(model.module.discriminator.parameters())))

    print('Training Starts!!!')
    model.train()
    for iteration, (images, targets) in enumerate(data_loader['train'], args.resume_iter+1):
        det_loss, dis_loss, loss_dict = model(images, targets)
    
        det_loss = det_loss.mean()
        dis_loss = dis_loss.mean()

        loss_dict = {k:v.mean().item() for k, v in loss_dict.items()}

        detector_train_flag = not (iteration > cfg.SOLVER.DETECTOR.INIT_TRAIN_ITER and iteration <= cfg.SOLVER.DETECTOR.INIT_TRAIN_ITER + cfg.SOLVER.DISCRIMINATOR.INIT_TRAIN_ITER)
        discriminator_train_flag = iteration > cfg.SOLVER.DETECTOR.INIT_TRAIN_ITER      

        ### Detector gradient calculation
        if detector_train_flag:
            optimizer.zero_grad()
            scaler.scale(det_loss).backward(retain_graph=discriminator_train_flag, inputs=detector_params)
            if cfg.SOLVER.DETECTOR.GRADIENT_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad, model.parameters()), cfg.SOLVER.DETECTOR.GRADIENT_CLIP)

        ### Discriminator gradient calculation
        if discriminator_train_flag:
            d_optimizer.zero_grad()
            scaler.scale(dis_loss).backward(inputs=discriminator_params)
            if cfg.SOLVER.DISCRIMINATOR.GRADIENT_CLIP > 0:
                scaler.unscale_(d_optimizer)
                torch.nn.uitls.clip_grad_norm_(filter(lambda p:p.requires_grad, model.parameters()), cfg.SOLVER.DISCRIMINATOR.GRADIENT_CLIP)

        ### Update
        if detector_train_flag:
            scaler.step(optimizer)
        if discriminator_train_flag:
            scaler.step(d_optimizer)

        scheduler.step()
        d_scheduler.step()

        scaler.update()

        # for param in model.module.extractors[0].parameters():
        #     print(param.mean())
        #     break

        for k in loss_dict.keys():
            if k in logging_losses:
                logging_losses[k] += loss_dict[k]
            else:
                logging_losses[k] = loss_dict[k]

        trained_time += time.time() - end
        end = time.time()

        ### Logging
        if iteration % args.log_step == 0:
            logging_losses = {k: v / args.log_step for k, v in logging_losses.items()}
            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            print('===> Iter: {:07d}, LR: {:.06f}, Cost: {:.02f}s, Eta: {}, Detector Loss: {:.6f}, Discriminator Loss: {:.6f}'.format(iteration, optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)), logging_losses['total_loss'], logging_losses['total_adv_loss']))

            if summary_writer is not None:
                if args.gradient_logging:
                    extractor_grad, detector_grad, discriminator_grad = 0, 0, 0
                    for param in model.module.extractors.parameters():
                        if param.grad is not None:
                            extractor_grad += LA.norm(param.grad)
                    for param in model.module.detector.parameters():
                        if param.grad is not None:
                            detector_grad += LA.norm(param.grad)
                    for param in model.module.discriminator.parameters():
                        if param.grad is not None:
                            discriminator_grad += LA.norm(param.grad)

                    logging_losses['extractor_grad'] = extractor_grad
                    logging_losses['detector_grad'] = detector_grad
                    logging_losses['discriminator_grad'] = discriminator_grad


                for k in logging_losses.keys():
                    summary_writer.add_scalar('train/{}'.format(k), logging_losses[k], global_step=iteration)
                summary_writer.add_scalar('train/detector_lr', optimizer.param_groups[0]['lr'], global_step=iteration)
                summary_writer.add_scalar('train/discriminator_lr', d_optimizer.param_groups[0]['lr'], global_step=iteration)

            logging_losses = {}

            tic = time.time()

        ### Save snapshot
        if iteration % args.save_step == 0 and not args.debug:
            model_path = os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(iteration))
            optimizer_path = os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(iteration))
            scaler_path = os.path.join(cfg.OUTPUT_DIR, 'scaler', 'iteration_{}.pth'.format(iteration))

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

            torch.save(model.module.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            torch.save(scaler.state_dict(), scaler_path)

            print('=====> Save Checkpoint to {}'.format(model_path))

        ### Validation
        if 'val' in data_loader.keys() and iteration % args.eval_step == 0:
            print('Validating...')
            model.eval()
            val_loss = 0
            for _ in tqdm(data_loader['val']):
                with torch.inference_mode():
                    loss = model()
                val_loss += loss.item()

            val_loss /= len(data_loader['val'])

            validation_time = time.time() - end
            trained_time += validation_time
            end = time.time()
            tic = time.time()
            print('======> Cost: {:2f}s, Loss: {:.06f}'.format(validation_time, val_loss))

            if summary_writer:
                summary_writer.add_scalar('val/loss', val_loss, global_step=iteration)

            model.train()


def do_pretrain_for_mp(args, cfg, model, optimizer, scheduler, d_optimizer, d_scheduler, data_loader, summary_writer):
    max_iter = len(data_loader['train']) + args.resume_iter
    trained_time = 0
    tic = time.time()
    end = time.time()

    logging_losses = {}

    print('Pretraining Starts!!!')
    model.train()
    for iteration, (images, targets) in enumerate(data_loader['train'], args.resume_iter+1):
        if iteration > 5000:
            print('Pretraining is completed.')
            break
        det_loss, dis_loss, loss_dict = model(images, targets, pretrain=True)
    
        det_loss = det_loss.mean()
        dis_loss = dis_loss.mean()

        loss_dict = {k:v.mean().item() for k, v in loss_dict.items()}

        detector_train_flag = not (iteration > 5000 and iteration <= 5000)
        discriminator_train_flag = iteration > 5000

        # print(detector_train_flag)
        
        ### Detector update
        if detector_train_flag:
            optimizer.zero_grad()
            det_loss.backward(retain_graph = discriminator_train_flag)
            if cfg.SOLVER.DETECTOR.GRADIENT_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad, model.parameters()), cfg.SOLVER.DETECTOR.GRADIENT_CLIP)
            optimizer.step()
        scheduler.step()

        ### Discriminator update
        if discriminator_train_flag:
            d_optimizer.zero_grad()
            dis_loss.backward()
            if cfg.SOLVER.DISCRIMINATOR.GRADIENT_CLIP > 0:
                torch.nn.uitls.clip_grad_norm_(filter(lambda p:p.requires_grad, model.parameters()), cfg.SOLVER.DISCRIMINATOR.GRADIENT_CLIP)
            d_optimizer.step()
        d_scheduler.step()

        for k in loss_dict.keys():
            if k in logging_losses:
                logging_losses[k] += loss_dict[k]
            else:
                logging_losses[k] = loss_dict[k]

        trained_time += time.time() - end
        end = time.time()

        ### Logging
        if iteration % args.log_step == 0:
            logging_losses = {k: v / args.log_step for k, v in logging_losses.items()}
            eta_seconds = int((trained_time / iteration) * (10000 - iteration))
            print('===> Iter: {:07d}, LR: {:.06f}, Cost: {:.02f}s, Eta: {}, Detector Loss: {:.6f}, Discriminator Loss: {:.6f}'.format(iteration, optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)), logging_losses['total_loss'], logging_losses['total_adv_loss']))

            # if summary_writer is not None:
            #     for k in logging_losses.keys():
            #         summary_writer.add_scalar('train/{}'.format(k), logging_losses[k], global_step=iteration)
            #     summary_writer.add_scalar('train/detector_lr', optimizer.param_groups[0]['lr'], global_step=iteration)
            #     summary_writer.add_scalar('train/discriminator_lr', d_optimizer.param_groups[0]['lr'], global_step=iteration)

            logging_losses = {}

            tic = time.time()
