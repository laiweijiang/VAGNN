import world
import dataloader
import model
from pprint import pprint

if world.dataset in ['wechat', 'tiktok', 'takatak','wechat2', 'tiktok2','takatak2']:
    dataset = dataloader.Loader(path="../data/" + world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'lgn': model.VAGNN
}