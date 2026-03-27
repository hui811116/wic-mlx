import mlx.core as mx
import mlx.nn as nn
import utils as uts

class Encoder(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dims, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, out_dims),
        )
    def __call__(self,x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, in_dims, out_dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(out_dims, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, in_dims),
        )
    def __call__(self,x):
        return self.net(x)


class NetworkWIC(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num):
        super(NetworkWIC, self).__init__()
        self.view = view
        self.input_size = input_size
        self.feature_dim = feature_dim
        self.high_feature_dim = high_feature_dim
        self.class_num = class_num
        self.encoders = [Encoder(input_size[i], feature_dim) for i in range(view)]
        #self.encoders = nn.ModuleList([Encoder(input_size[i], feature_dim) for i in range(view)])
        self.decoders = [Decoder(input_size[i], feature_dim) for i in range(view)]
        #self.decoders = nn.ModuleList([Decoder(feature_dim, input_size[i]) for i in range(view)])
        #self.cluster_layer = nn.Linear(feature_dim, class_num)
        self.cluster_layer = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(),)

    def __call__(self, xvs):
        # For conditional independent  views
        views = len(xvs)
        h_views, q_views, z_views = [], [], []
        xrs = []
        for v in range(views):
            x = xvs[v]
            h = self.encoders[v](x)
            q = self.cluster_layer(h)
            z = self.decoders[v](h)
            xr = self.decoders[v](h)
            h_views.append(h)
            q_views.append(q)
            z_views.append(z)
            xrs.append(xr)
        return h_views, q_views, xrs, z_views
        