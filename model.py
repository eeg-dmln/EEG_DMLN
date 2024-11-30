import itertools

import torch
from EasyPro.disp_tool import progress_for

from TheVirtualBrain.activity_analyse import normalize
from .basic import Module, ResConv, ResTConv, MGU
from torch.nn import Linear
from TheVirtualBrain import sensor, connection
target_region = connection.Connection.from_file().cortical
target_region[-16:] = True


class STVAE(Module):
    def __init__(self, eeg_channel_dim=60, latent_dim=5 * 7 * 9, internal_activity_dim=210):
        """
        时空变分自编码器。
        :param eeg_channel_dim: 脑电信号的维数
        :param latent_dim: 潜在状态个数
        :param internal_activity_dim: 脑内部神经活动维数
        """
        super().__init__()

        self.latent_dim = latent_dim

        self.spatial_encoder = ResConv(1, 5, 5, 5, 3, 3)
        self.temporal_encoder = MGU(input_size=latent_dim, num_layers=2, bidirectional=True, hidden_size=latent_dim * 2,
                                    batch_first=True)
        self.distribution_readout = Linear(latent_dim * 4, latent_dim * 2)

        self.internal_activity_decoder = ResTConv(5, 5, 3, 3, 5, 5)
        self.internal_activity_readout = Linear(latent_dim, internal_activity_dim)
        self.eeg_decoder = ResTConv(5, 1, 3, 3, 5, 5)

        self.unsupervised = 0
        self.supervised = 0

    def forward(self, din, reconstruct_mode=True):
        temporal_latent, temporal_latent_var = self.latent(din)
        if reconstruct_mode:
            return self.reconstruct(temporal_latent, temporal_latent_var)
        return self.estimate(temporal_latent, temporal_latent_var)

    def din_preprocess(self, din: torch.Tensor):
        din = din.to(self.device)

        if len(din.size()) == 2:
            din = din.unsqueeze(0)
        # 批次*时间*导联

        din = sensor.matrix_output(din).unsqueeze(2)
        return din

    def latent(self, din):
        """

        :param din: EEG信号 【时间*批次*导联】
        :return:
        """

        din = self.din_preprocess(din)

        spatial_feature = self.spatial_encoder(din)

        dim = list(spatial_feature.size())[-3:]
        temporal_feature, _ = self.temporal_encoder(spatial_feature.flatten(2))
        temporal_feature = self.distribution_readout(temporal_feature)

        temporal_latent = temporal_feature[..., ::2].unflatten(-1, dim)
        temporal_latent_var = temporal_feature[..., 1:][..., ::2].unflatten(-1, dim)

        return temporal_latent, temporal_latent_var

    def reconstruct(self, temporal_latent, temporal_latent_var):
        # manifold sampling
        if self.training:
            temporal_latent = temporal_latent + (2 * torch.rand_like(temporal_latent_var) - 1) * temporal_latent_var

        reconstructed_eeg = self.eeg_decoder(temporal_latent)
        reconstructed_eeg = sensor.original_output(reconstructed_eeg.squeeze(-3))

        return reconstructed_eeg

    def estimate(self, temporal_latent, temporal_latent_var):
        # manifold sampling
        if self.training:
            temporal_latent = temporal_latent + (2 * torch.rand_like(temporal_latent_var) - 1) * temporal_latent_var

        internal_activity = self.internal_activity_decoder(temporal_latent)
        internal_activity = self.internal_activity_readout(internal_activity.flatten(-3))

        return internal_activity
    @property
    def name(self):
        return f'unsupervised{self.unsupervised}_supervised_{self.supervised}', 'STVAE'



def use(model, original_eeg):
    model.train(False)

    original_eeg = original_eeg.to(model.device)

    with torch.no_grad():
        temporal_latent = model.latent(original_eeg)

        reconstructed_eeg = model.eeg_decoder(temporal_latent)
        reconstructed_eeg = sensor.original_output(reconstructed_eeg.squeeze(-3))

        internal_activity = model.internal_activity_decoder(temporal_latent)
        internal_activity = model.internal_activity_readout(internal_activity.flatten(-3))

    return reconstructed_eeg, internal_activity, temporal_latent

def epoch_for(epoch_num, data_loader):
    for i in range(epoch_num):
        for batch_data in data_loader:
            yield batch_data
            break

def total_train(data_loader, model: STVAE, optimizer, loss_function, epoch_num,
                       validate_num=1, save_interval=50, use_gpu=True):
    # print('start synthetic training')
    supervise_train_loss = []
    supervise_test_loss = []
    unsupervise_train_loss = []
    unsupervise_test_loss = []

    for batch_data in progress_for(epoch_for(epoch_num, data_loader), epoch_num * len(data_loader)):
        _, ai, eeg = batch_data

        eeg = eeg.transpose(0, 1)
        ai = ai.transpose(0, 1)

        # region normalize
        # endregion

        train_eeg = eeg[validate_num:]
        test_eeg = eeg[:validate_num]
        train_ai = ai[validate_num:][..., target_region]
        test_ai = ai[:validate_num][..., target_region]

        if use_gpu:
            train_eeg = train_eeg.cuda()
            test_eeg = test_eeg.cuda()
            train_ai = train_ai.cuda()
            test_ai = test_ai.cuda()

        model.train(True)

        # unsupervised training
        temporal_latent, temporal_latent_var = model.latent(train_eeg)
        reconstructed_eeg = model.reconstruct(temporal_latent, temporal_latent_var)
        reconstructed_loss = loss_function(reconstructed_eeg, train_eeg, temporal_latent, temporal_latent_var)

        optimizer.zero_grad()
        reconstructed_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.unsupervised += 1
        unsupervise_train_loss.append(reconstructed_loss.item())
        # print('unsupervised train loss', reconstructed_loss.item())


        # supervised training
        temporal_latent, temporal_latent_var = model.latent(train_eeg)
        estimated_ai = model.estimate(temporal_latent, temporal_latent_var)
        estimated_loss = loss_function(estimated_ai, train_ai, temporal_latent, temporal_latent_var)

        optimizer.zero_grad()
        estimated_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.supervised += 1
        supervise_train_loss.append(estimated_loss.item())
        # print('supervised train loss', estimated_loss.item())

        with torch.no_grad():
            model.train(False)
            temporal_latent, temporal_latent_var = model.latent(test_eeg)
            estimated_ai = model.estimate(temporal_latent, temporal_latent_var)
            reconstructed_eeg = model.reconstruct(temporal_latent, temporal_latent_var)

            estimated_loss = loss_function(estimated_ai, test_ai, temporal_latent, temporal_latent_var)
            reconstructed_loss = loss_function(reconstructed_eeg, test_eeg, temporal_latent, temporal_latent_var)
            supervise_test_loss.append(estimated_loss.item())
            unsupervise_test_loss.append(reconstructed_loss.item())
            # print('test loss', estimated_loss.item())

        if model.supervised % save_interval == 0:
            yield (
                model,
                torch.Tensor(supervise_train_loss).mean(),
                torch.Tensor(unsupervise_train_loss).mean(),
                torch.Tensor(supervise_test_loss).mean(),
                torch.Tensor(unsupervise_test_loss).mean(),
            )

def unsupervised_train(data_loader, model: STVAE, loss_function, train_loss, test_loss, epoch_num,
                       validate_num=1, save_interval=50, use_gpu=True):
    optimizer = torch.optim.Adadelta(model.parameters(True))
    print('start unsupervised training')
    for batch_data in progress_for(epoch_for(epoch_num, data_loader), epoch_num * len(data_loader)):
        _, _, eeg = batch_data
        eeg = eeg.transpose(0, 1)
        train_eeg = eeg[validate_num:]
        test_eeg = eeg[:validate_num]
        if use_gpu:
            train_eeg = train_eeg.cuda()
            test_eeg = test_eeg.cuda()

        model.train(True)
        reconstructed_eeg, _ = model.reconstruct(train_eeg)
        reconstructed_loss = loss_function(reconstructed_eeg, train_eeg)
        optimizer.zero_grad()
        reconstructed_loss.backward()
        optimizer.step()
        model.unsupervised += 1
        train_loss.append(reconstructed_loss.item())
        print('train loss', reconstructed_loss.item())

        with torch.no_grad():
            model.train(False)
            reconstructed_eeg, _ = model.reconstruct(test_eeg)
            reconstructed_loss = loss_function(reconstructed_eeg, test_eeg)
            test_loss.append(reconstructed_loss.item())
            print('test loss', reconstructed_loss.item())

        if model.unsupervised % save_interval == 0:
            yield model, train_loss, test_loss

def supervised_train(data_loader, model: STVAE, loss_function, train_loss, test_loss, epoch_num,
                     validate_num=1, save_interval=50, use_gpu=True):
    optimizer = torch.optim.Adadelta(model.parameters(False))
    print('start supervised training')
    for batch_data in progress_for(epoch_for(epoch_num, data_loader), epoch_num * len(data_loader)):
        _, ai, eeg = batch_data

        eeg = eeg.transpose(0, 1)
        ai = ai.transpose(0, 1)

        train_eeg = eeg[validate_num:]
        test_eeg = eeg[:validate_num]
        train_ai = ai[validate_num:][..., target_region]
        test_ai = ai[:validate_num][..., target_region]

        if use_gpu:
            train_eeg = train_eeg.cuda()
            test_eeg = test_eeg.cuda()
            train_ai = train_ai.cuda()
            test_ai = test_ai.cuda()

        model.train(True)
        estimated_ai, _ = model(train_eeg)
        estimated_loss = loss_function(estimated_ai, train_ai)
        optimizer.zero_grad()
        estimated_loss.backward()
        optimizer.step()
        model.supervised += 1
        train_loss.append(estimated_loss.item())
        print('train loss', estimated_loss.item())

        with torch.no_grad():
            model.train(False)
            estimated_ai, _ = model(test_eeg)
            estimated_loss = loss_function(estimated_ai, test_ai)
            test_loss.append(estimated_loss.item())
            print('test loss', estimated_loss.item())

        if model.supervised % save_interval == 0:
            yield model, train_loss, test_loss