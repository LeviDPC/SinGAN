import torchaudio
import torch
import julius


class AudioSample:

    def __init__(self, opt, path, clone=False, data=None, sr=None):
        self.opt = opt
        self.path = path
        if not clone:
            if opt.audio_backend == "soundfile":
                self.data, self.sr = torchaudio.load(path, normalization=opt.norm)
            elif opt.audio_backend == "sox_io":
                self.data, self.sr = torchaudio.load(path, normalize=opt.norm)
            if not opt.not_cuda and torch.cuda.is_available():
                self.data = self.data.to(torch.device('cuda'))

            # Data is stored in a tensor of shape 1 (batch size) x audio channels x width (floats in channel)
            self.data = self.data[0].view(1, 1, -1)
            if self.sr != sr:
                self.resample_to_julius(sr)

            if self.data.shape[2] % 2 != 0 and opt.make_input_tensor_even > 0:
                self.data=self.data[:,:,0:self.data.shape[2]-1]
        else:
            self.data = data
            self.sr = sr

    @staticmethod
    def static_save(data_in, sr_in, path_in):
        data_in = data_in.to(torch.device('cpu'))
        torchaudio.save(path_in, data_in.view(1, -1), sr_in, 32)

    @staticmethod
    def static_resample(data_in, old_sr, new_sr):
        saved_shape = data_in.shape
        if saved_shape[1] == 1:
            out = (torchaudio.transforms.Resample(orig_freq=old_sr, new_freq=new_sr)(data_in)[0, 0, :]).view(1, 1, -1)
        # Things are mostly only setup for one channel. This is the exception
        elif saved_shape[1] == 2:
            chan1 = (torchaudio.transforms.Resample(orig_freq=old_sr, new_freq=new_sr)(data_in)[0, 0, :]).view(1, 1, -1)
            chan2 = (torchaudio.transforms.Resample(orig_freq=old_sr, new_freq=new_sr)(data_in)[0, 1, :]).view(1, 1, -1)
            out = torch.cat(chan1, chan2, dim=1)
        return out


    @staticmethod
    def static_pad(data_in, pad_each_side_by, opt):
        if opt.pad_with_noise < 1:
            part1 = torch.zeros(1, data_in.shape[1], pad_each_side_by, dtype=torch.float32)
            part2 = part1
        else:
            if opt.not_cuda > 0:
                device = 'cpu'
            else:
                device = 'cuda'

            part1 =torch.randn(1, data_in.shape[1], pad_each_side_by, device=device)
            part2 = torch.randn(1, data_in.shape[1], pad_each_side_by, device=device)
        data_in = torch.cat((part1, data_in, part2), dim=2)
        # data_in=torch.cat((data_in, part1), dim=2)
        return data_in

    @staticmethod
    def resample_to_julius_static(data_in, sr_in, new_sr):
        return julius.resample_frac(data_in, sr_in, new_sr)

    def resample_to_julius(self, target):
        self.data = julius.resample_frac(self.data, self.sr, target)
        self.sr = target

    def resample_by(self, scale):

        saved_shape = self.data.shape
        if saved_shape[1] == 1:
            newsr = int(self.sr * scale)
            self.data = (torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=newsr)(self.data)[0, 0, :]).view(1,
                                                                                                                     1,
                                                                                                                     -1)
        # Things are mostly only setup for one channel. This is the exception
        elif saved_shape[1] == 2:
            newsr = int(self.sr * scale)
            chan1 = (torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=newsr)(self.data)[0, 0, :]).view(1, 1,
                                                                                                                 -1)
            chan2 = (torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=newsr)(self.data)[0, 1, :]).view(1, 1,
                                                                                                                 -1)
            self.data = torch.cat(chan1, chan2, dim=1)
        self.sr = newsr

    def resample_by_sr(self, sr_in):
        saved_shape = self.data.shape
        if saved_shape[1] == 1:
            newsr = sr_in
            self.data = (torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=newsr)(self.data)[0, 0, :]).view(1,
                                                                                                                     1,
                                                                                                                     -1)
        # Things are mostly only setup for one channel. This is the exception
        elif saved_shape[1] == 2:
            newsr = sr_in
            chan1 = (torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=newsr)(self.data)[0, 0, :]).view(1, 1,
                                                                                                                 -1)
            chan2 = (torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=newsr)(self.data)[0, 1, :]).view(1, 1,
                                                                                                                 -1)
            self.data = torch.cat(chan1, chan2, dim=1)
        self.sr = newsr

    @staticmethod
    def static_normalize(data_in):

        factor=torch.max(torch.abs(data_in))
        if factor > 1:
            out = data_in/factor
            return out
        else:
            return data_in

    def clone(self):
        return AudioSample(self.opt, self.path, clone=True, data=self.data, sr=self.sr)

    def write_to_file(self, path):
        torchaudio.save(path, self.data, self.sr)
