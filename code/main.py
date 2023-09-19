import world
import utils
from world import cprint
import torch
import time
import Procedure

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
procedure=Procedure.Procedure(Recmodel, world.config)
weight_file = utils.getFileName()


best_hr, best_ndcg = 0, 0
best_epoch = 0
count, epoch = 0, 0

while count < 10:
    start = time.time()
    output_information = procedure.BPR_train_original(dataset, Recmodel, neg_k=world.config['neg'])
    cprint("[valid]")
    res = Procedure.Test(dataset, Recmodel, 'valid', world.config['multicore'])
    hr1, ndcg1 = res['recall'][0], res['ndcg'][0]
    hr2, ndcg2 = res['recall'][0], res['ndcg'][0]
    print(f'EPOCH[{epoch + 1}/{world.TRAIN_epochs}] {output_information}')
    if hr1 > best_hr :
        best_epoch = epoch
        count = 0
        best_hr, best_ndcg = hr1, ndcg1
        model_dir = weight_file + str(epoch) + '.model'
        torch.save(Recmodel.state_dict(), model_dir)
    else:
        # 小于10次退出训练
        count += 1
    epoch += 1
print("End. Best epoch {:03d}: HR = {:.4f}, NDCG = {:.4f} in invalid data".format(
    best_epoch, best_hr, best_ndcg))
print("save to"+model_dir)

# test
Recmodel.load_state_dict(torch.load(weight_file + str(best_epoch) + '.model'))
cprint("[test]")
res = Procedure.Test(dataset, Recmodel, 'test', world.config['multicore'])
