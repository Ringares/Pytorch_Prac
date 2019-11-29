import torch

from model.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))


if __name__ == '__main__':
    """
    
    x : (4,5)
    l = x->linear(1) : (4,1)
    embed = x->embedding(16) : (4,5,16) 
    fm = embed->fm(reduce_sum) : (4,1) # sum[sum(4,5,16, dim=1), sum(4,5,16, dim=1), dim=1, keep_dim=True]
    y = sigmoid(squeeze(l+fm, 1)) : (4)
    
    """
    field_dims = [3, 4, 5, 6, 3]
    x = torch.randint(1, 3, (4, 5))
    model = FactorizationMachineModel(field_dims, 16)
    print(model(x))
