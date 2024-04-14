from architectures import FFN
def ffn(n, **kwargs):
    model = FFN(n, 100, **kwargs)
    return model