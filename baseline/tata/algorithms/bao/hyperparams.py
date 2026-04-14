params = {
    'imdb' : {
        'bs' : 128,
        'epochs' : 200,
        'lr' : 1e-4, 
        'hid' : 64,
        'in_channel' : 12,
    },
    'stats' : {
        'bs' : 64,
        'epochs' : 160,
        'lr' : 1e-4, 
        'hid' : 64,
        'in_channel' : 12,
    }
}

def get_hyperparams(dataset):
    return params[dataset]