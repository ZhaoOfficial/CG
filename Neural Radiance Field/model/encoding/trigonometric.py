import torch

from encoding import Encoder 

class TrihonometricEncoder(Encoder):
    """
    Also called positional encoder. Using a series of sin and cos functions as encoding functions.
    sin(2 ** 0 * x), sin(2 ** 1 * x), ..., sin(2 ** L * x)
    cos(2 ** 0 * x), cos(2 ** 1 * x), ..., cos(2 ** L * x)
    """

    def __init__(self, config: dict):
        self.in_dim = config["in_dim"]
        self.num_frequencies = config["num_frequencies"]
        self.out_dim = self.in_dim * self.num_frequencies * 2

        # frequency.size() = (self.num_frequencies)
        if config["log_sampling"]:
            # equidistant in logarithm space
            self.frequency = 2.0 ** torch.linspace(0.0, float(self.num_frequencies - 1), self.num_frequencies)
        else:
            # equidistant in linear space
            self.frequency = torch.linspace(0.0, 2.0 ** float(self.num_frequencies - 1), self.num_frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size() = (N, self.in_dim)
        y = []
        for i in range(self.in_dim):
            for fn in [torch.sin, torch.cos]:
                y.append(fn(x[..., i: i + 1] * self.frequency))
        # y.size() = (N, self.out_dim)
        y = torch.cat(y, dim=-1)
        return y

class Embedder:
    def __init__(self):
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = 3
        out_dim = 0
            
        max_freq = 10
        N_freqs = 10
        
        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

if __name__ == "__main__":
    from ioutil import JsonIO
    embed = Embedder()
    trigo = TrihonometricEncoder(JsonIO.input(r"D:\Mywork\Computer Graphics\Neural Radiance Field\config\hotdog.json")["encoding"])

    a = torch.Tensor([[[1,2,3], [4,5,6]]], dtype=torch.float32)
    b = embed.embed(a)
    c = trigo(a)
    print(torch.allclose(b, c))
