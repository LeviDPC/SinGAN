import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import wandb
import matplotlib.pyplot as plt
from SinGAN.AudioSample import AudioSample
from SinGAN.imresize import imresize


def train(opt, real, Gs, Zs, reals, NoiseAmp):
    in_s = 0
    scale_num = 0

    # real.resample_by(opt.scale1)
    real.resample_to_julius(opt.SR_pyr[-1])
    print("new shape", real.data.shape)

    # reals, sr_list = functions.creat_reals_pyramid_audio(real, reals, opt, verbose=False)
    reals = functions.creat_reals_pyramid(real, reals, opt, verbose=False)
    sr_list = opt.SR_pyr
    nfc_prev = 0

    for x in reals:
        print(x.device)

    while scale_num < opt.stop_scale + 1:
        if opt.change_channel_count > 0:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)

        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        # plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        # plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        # plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)
        if opt.alt_pyramid_exp > 0:
            level=len(Gs)
            opt.ker_size=opt.ker_size_pyr[level]

        D_curr, G_curr = init_models(opt)
        if (nfc_prev == opt.nfc and opt.alt_pyramid_exp == 0):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num - 1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))

        z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals, sr_list, Gs, Zs, in_s, NoiseAmp, opt)

        G_curr = functions.reset_grads(G_curr, False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr, False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num += 1
        nfc_prev = opt.nfc
        del D_curr, G_curr
    return

def train_on_audio_resume(opt, real, Gs, Zs, reals, NoiseAmp, in_s):
    scale_num = opt.level_to_resume_at

    # real.resample_by(opt.scale1)
    #real.resample_to_julius(opt.SR_pyr[-1])
    #print("new shape", real.data.shape)

    # reals, sr_list = functions.creat_reals_pyramid_audio(real, reals, opt, verbose=False)
    #reals = functions.creat_reals_pyramid_julius(real, reals, opt, verbose=False)
    sr_list = opt.SR_pyr
    nfc_prev = 0

    for x in reals:
        print(x.device)

    while scale_num < opt.stop_scale + 1:
        if opt.change_channel_count > 0:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_, scale_num)

        try:
            os.makedirs(opt.outf)
        except OSError:
            pass

        # plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        # plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        # plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

        D_curr, G_curr = init_models(opt)
        #TODO: This next line doesn't work for the first scale on resume if change_channel_count is > 0
        if (nfc_prev == opt.nfc) or opt.change_channel_count==0:
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_, scale_num - 1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_, scale_num - 1)))

        z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals, sr_list, Gs, Zs, in_s, NoiseAmp, opt)

        G_curr = functions.reset_grads(G_curr, False)
        G_curr.eval()
        D_curr = functions.reset_grads(D_curr, False)
        D_curr.eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num += 1
        nfc_prev = opt.nfc
        del D_curr, G_curr
    return

# On Laptop 100 epochs at first scale takes about 0:49 on debugger. Second scale: On 100 epochs at about 4:45
def train_single_scale(netD, netG, reals, sr_list, Gs, Zs, in_s, NoiseAmp, opt, centers=None):
    curr_scale = len(Gs)
    if curr_scale == 0:
        print("It's 0!")
    if curr_scale == 1:
        print("It's 1!")
    if curr_scale == 2:
        print("It's 2!")

    update_count = 1

    curr_sr = sr_list[curr_scale]
    real = reals[curr_scale]
    opt.nzx = real.shape[2]  # +(opt.ker_size-1)*(opt.num_layer)

    opt.nz = real.shape[0]
    opt.receptive_field = int(opt.ker_size) + ((int(opt.ker_size) - 1) * (int(opt.num_layer) - 1)) * int(opt.stride)

    # We aren't going to worry about padding. It's easier to just make it 0 (i.e. do nothing) then
    # completely remove it
    # pad_noise = 0 #int(((opt.ker_size - 1) * opt.num_layer) / 2)
    # pad_image = 0 #int(((opt.ker_size - 1) * opt.num_layer) / 2)


    pad_num = int((opt.ker_size - 1)  * opt.dilation * opt.num_layer  * (1/2))




    # Was an if then for animation train here I removed

    alpha = opt.alpha

    fixed_noise = functions.generate_noise([opt.nc_z, opt.nzx], device=opt.device)
    # This next line updated to bring it inline with later versions of pytorch by
    # Levi Pfantz on 10/20/2020
    z_opt = torch.full(fixed_noise.shape, 0, dtype=torch.float32, device=opt.device)
    z_opt = AudioSample.static_pad(z_opt, pad_num, opt)


    # setup optimizer
    if len(Gs) == 3 and opt.adjust_after_levels > 0:
        opt.lr_d = 0.0005
    if len(Gs) == 5 and opt.adjust_after_levels > 0:
        opt.lr_d = 0.0004
    if len(Gs) == 7 and opt.adjust_after_levels > 0:
        opt.lr_d = 0.0003
    if len(Gs) == 9 and opt.adjust_after_levels > 0:
        opt.lr_d = 0.0002
    if len(Gs) == 11 and opt.adjust_after_levels > 0:
        opt.lr_d = 0.0001

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[1600], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1600], gamma=opt.gamma)

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []


    final_GLoss_last = 100
    final_GLoss = 0

    # On Desktop cpu 100 epochs takes about 2:17
    # on 300 epochs in about 14:45
    for epoch in range(opt.niter):
        # If this is the first epoch of the first level then z_opt needs to be defined?
        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt = functions.generate_noise([1, opt.nzx], device=opt.device)
            z_opt = z_opt.expand(1, 1, opt.nzx)
            z_opt = AudioSample.static_pad(z_opt, pad_num, opt)
            noise_ = functions.generate_noise([1, opt.nzx], device=opt.device)
            noise_ = noise_.expand(1, 1, opt.nzx)
            noise_ = AudioSample.static_pad(noise_, pad_num, opt)
        else:
            noise_ = functions.generate_noise([opt.nc_z, opt.nzx], device=opt.device)
            noise_ = AudioSample.static_pad(noise_, pad_num, opt)


        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.steps):
            # train with real
            netD.zero_grad()
            # Very First time on Laptop takes:
            output = netD(real).to(opt.device)
            # D_real_map = output.detach()
            errD_real = -output.mean()  # -a
            if opt.smooth_real_labels > 0:
                print("smooth")
                real_mod=((torch.rand(1)*0.5)+0.7)
                errD_real = errD_real*real_mod
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fake
            # If this is the first step of the first epoch
            if (j == 0) & (epoch == 0):
                # If this is the first level
                if (Gs == []) & (opt.mode != 'SR_train'):
                    # We need to setup a previous of random noise since there wasn't a previous level do derive from
                    prev = torch.full([1, opt.nc_z, opt.nzx], 0, dtype=torch.float32, device=opt.device)
                    in_s = prev
                    prev = AudioSample.static_pad(prev, pad_num, opt)
                    # This next line updated to bring it inline with later versions of pytorch by
                    # Levi Pfantz on 10/20/2020
                    z_prev = torch.full([1, opt.nc_z, opt.nzx], 0, dtype=torch.float32, device=opt.device)
                    z_prev = AudioSample.static_pad(z_prev, pad_num, opt)
                    opt.noise_amp = 1

                # removed an if else for SR

                else:
                    # Because SinGAN calculates two different losses (adversarial term and reconstruction)
                    # We will generate two new images with the past scale's generator and concat them with noise
                    # The result is prev for adversarial and z_prev for reconstruction
                    prev = draw_concat(Gs, Zs, reals, sr_list, NoiseAmp, in_s, 'rand',
                                       opt)  # Possible issue in draw concat audio
                    prev = AudioSample.static_pad(prev, pad_num, opt)
                    z_prev = draw_concat(Gs, Zs, reals, sr_list, NoiseAmp, in_s, 'rec', opt)

                    criterion = nn.MSELoss()
                    # RMSE = Root mean squared error
                    # Calculate RMSE, We'll use it when we calculate the reconstruction loss
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = AudioSample.static_pad(z_prev, pad_num, opt)
            else:
                prev = draw_concat(Gs, Zs, reals, sr_list, NoiseAmp, in_s, 'rand', opt)
                prev = AudioSample.static_pad(prev, pad_num, opt)

            # removed an if for paint train

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp * noise_ + prev

            fake = netG(noise.detach(), prev)
            if opt.normalize_generator_output > 0:
                prenorm_max=torch.max(fake)
                prenorm_min=torch.min(fake)
                fake = AudioSample.static_normalize(fake)
            output = netD(fake.detach())
            errD_fake = output.mean()
            if opt.smooth_fake_labels > 0:
                fake_mod=((torch.rand(1)*0.5)+0.7)
                errD_fake = errD_fake*fake_mod
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            temp = errD

            if opt.update_only_with_lower_Gloss == 0 or final_GLoss_last >= final_GLoss:
                if update_count % int(opt.steps_to_update) == 0 or int(opt.update_every_x) == 0:
                    optimizerD.step()
                update_count += 1

            ############################
            # (2) Update G network: maximize D(G(z))
            ###########################

            netG.zero_grad()

            output = netD(fake)
            # D_fake_map = output.detach()
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha != 0:
                if opt.use_MAE > 0:
                    loss = nn.L1Loss()
                else:
                    loss = nn.MSELoss()

                # if then for paint_train removed

                Z_opt = opt.noise_amp * z_opt + z_prev
                rec_loss = alpha * loss(netG(Z_opt.detach(), z_prev), real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            if opt.update_only_with_lower_Gloss > 0:
                if update_count > 2:
                    final_GLoss_last=final_GLoss
                final_GLoss=errG.detach() + rec_loss


            optimizerG.step()

        errD2plot.append(errD.detach())
        errG2plot.append(errG.detach() + rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)
        if int(opt.wandb) > 0:
            if opt.normalize_generator_output>0:
                wandb.log({("prenom max at scale: " + str(len(Gs))): prenorm_max})
                wandb.log({("prenom min at scale: " + str(len(Gs))): prenorm_min})
            wandb.log({("errD at scale: " + str(len(Gs))): errD.detach()})
            wandb.log({("errG at scale: " + str(len(Gs))): errG.detach() + rec_loss})
            wandb.log({("D_real2plot at scale: " + str(len(Gs))): D_x})
            wandb.log({("D_fake2plot at scale: " + str(len(Gs))): D_G_z})
            wandb.log({("z_opt2pot at scale: " + str(len(Gs))): rec_loss})

        if epoch % 25 == 0 or epoch == (opt.niter - 1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))



        if epoch % 500 == 0 or epoch == (opt.niter - 1):
            # plt.imsave('%s/fake_sample.png' % (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            z_opt_to_save=netG(Z_opt.detach(), z_prev).detach()

            if opt.normalize_before_saving > 1:
                fake=AudioSample.static_normalize(fake)
                z_opt_to_save = AudioSample.static_normalize(z_opt_to_save)

            if opt.save_fake_progression > 0:
                if epoch==0 and not os.path.exists('%s/savedProgression/' % (opt.outf)):
                    os.makedirs('%s/savedProgression/' % (opt.outf))
                string=opt.outf+'/savedProgression/'+str(epoch)+'.wav'
                AudioSample.static_save(fake.detach(), curr_sr, string)

            AudioSample.static_save(fake.detach(), curr_sr, '%s/fake_sample.wav' % (opt.outf))
            # plt.imsave('%s/G(z_opt).png' % (opt.outf), functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            AudioSample.static_save(z_opt_to_save, curr_sr, '%s/G(z_opt).wav' % (opt.outf))

            # plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            # plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            # plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            # plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            # plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            # plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)

            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        if opt.use_schedulers > 0:
            schedulerD.step()
            schedulerG.step()

    functions.save_networks(netG, netD, z_opt, opt)
    return z_opt, in_s, netG


def draw_concat(Gs, Zs, reals, sr_list, NoiseAmp, in_s, mode, opt):
    # This function generates output from the previous scale that is up-scaled for the current scale
    # It does this by starting at the bottom of the pyramid and working it's way up. At each level it
    # uses the already trained generators to create output, upscale it and feed it to the next level.
    # The first option rand (random presumably) generates it's own random noise maps at each level which
    # should result in an entirely new, random image. The Next option reconstruction (presumably) uses
    # the saved noise maps for each level which (if I understand correctly) allows for a recreation of the
    # image previously used. It is what is used for calculating the reconstruction loss.
    G_z = in_s

    pad_num = int(((opt.ker_size - 1) * opt.num_layer) / 2) * opt.dilation

    if len(Gs) > 0:
        # This mode generates it's own noise
        if mode == 'rand':
            count = 0
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G, Z_opt, real_curr, real_next, noise_amp in zip(Gs, Zs, reals, reals[1:], NoiseAmp):

                if opt.alt_pyramid_exp > 0:
                    level = count
                    calc_ker_size=opt.ker_size_pyr[level]
                    pad_num = int(((opt.ker_size - 1) * opt.num_layer) / 2) * opt.dilation

                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_num], device=opt.device)
                    z = z.expand(1, 1, z.shape[2])
                else:
                    z = functions.generate_noise([opt.nc_z, Z_opt.shape[2] - 2 * pad_num], device=opt.device)
                z = AudioSample.static_pad(z, pad_num, opt)
                G_z = G_z[:, :, 0:real_curr.shape[2]]
                G_z = AudioSample.static_pad(G_z, pad_num, opt)
                z_in = noise_amp * z + G_z
                G_z = G(z_in.detach(), G_z)
                if opt.normalize_generator_output >0:
                    G_z=AudioSample.static_normalize(G_z)
                G_z = AudioSample.resample_to_julius_static(G_z, sr_list[count], sr_list[count + 1])

                if opt.adjust_upsampled > 0:
                    if G_z.shape[2] < real_next.shape[2]:
                        dif = real_next.shape[2] - G_z.shape[2]
                        G_z = torch.cat((G_z, torch.zeros([1,1,dif], dtype=torch.float32)), dim=2)

                # G_z = imresize(G_z, 1 / opt.scale_factor, op   t)
                G_z = G_z[:, :, 0:real_next.shape[2]]
                count += 1
        # This function uses Z_opt for noise
        if mode == 'rec':
            count = 0
            for G, Z_opt, real_curr, sr_curr, real_next, noise_amp in zip(Gs, Zs, reals, sr_list, reals[1:], NoiseAmp):

                if opt.alt_pyramid_exp > 0:
                    level = count
                    calc_ker_size=opt.ker_size_pyr[level]
                    pad_num = int(((opt.ker_size - 1) * opt.num_layer) / 2) * opt.dilation

                G_z = G_z[:, :, 0:real_curr.shape[2]]
                G_z = AudioSample.static_pad(G_z, pad_num, opt)
                z_in = noise_amp * Z_opt + G_z
                G_z = G(z_in.detach(), G_z)
                if opt.normalize_generator_output >0:
                    G_z=AudioSample.static_normalize(G_z)
                G_z = AudioSample.resample_to_julius_static(G_z, sr_list[count], sr_list[count + 1])

                if opt.adjust_upsampled > 0:
                    if G_z.shape[2] < real_next.shape[2]:
                        dif = real_next.shape[2] - G_z.shape[2]
                        G_z = torch.cat((G_z, torch.zeros([1,1,dif], dtype=torch.float32)), dim=2)

                # G_z = imresize(G_z, 1 / opt.scale_factor, opt)
                G_z = G_z[:, :, 0:real_next.shape[2]]
                # Lines that were commented out in the original function removed here
                count += 1
    return G_z



def train_paint(opt, Gs, Zs, reals, NoiseAmp, centers, paint_inject_scale):
    in_s = torch.full(reals[0].shape, 0, dtype=torch.float32, device=opt.device)
    scale_num = 0
    nfc_prev = 0

    while scale_num < opt.stop_scale + 1:
        if scale_num != paint_inject_scale:
            scale_num += 1
            nfc_prev = opt.nfc
            continue
        else:
            opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
            opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

            opt.out_ = functions.generate_dir2save(opt)
            opt.outf = '%s/%d' % (opt.out_, scale_num)
            try:
                os.makedirs(opt.outf)
            except OSError:
                pass

            # plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
            # plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
            plt.imsave('%s/in_scale.png' % (opt.outf), functions.convert_image_np(reals[scale_num]), vmin=0, vmax=1)

            D_curr, G_curr = init_models(opt)

            z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals[:scale_num + 1], Gs[:scale_num],
                                                      Zs[:scale_num], in_s, NoiseAmp[:scale_num], opt, centers=centers)

            G_curr = functions.reset_grads(G_curr, False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr, False)
            D_curr.eval()

            Gs[scale_num] = G_curr
            Zs[scale_num] = z_curr
            NoiseAmp[scale_num] = opt.noise_amp

            torch.save(Zs, '%s/Zs.pth' % (opt.out_))
            torch.save(Gs, '%s/Gs.pth' % (opt.out_))
            torch.save(reals, '%s/reals.pth' % (opt.out_))
            torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

            scale_num += 1
            nfc_prev = opt.nfc
        del D_curr, G_curr
    return


def init_models(opt):
    # generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)


    # discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)


    return netD, netG
