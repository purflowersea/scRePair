import torch
import tqdm
from torch.optim import Adam
from time import time

from utils import *
from encoder import *
from scRePair import scRePair
from data_loader import load_data

import torch.nn.functional as F
import csv
import json
from pathlib import Path

def pretrain_ae(model, x):
    print("Pretraining AE...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.rec_epoch)):
        z = model.encoder(x)
        x_hat = model.decoder(z)
        loss = F.mse_loss(x_hat, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def pretrain_gae(model, x, adj):
    print("Pretraining GAE...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.rec_epoch)):
        z, a = model.encoder(x, adj)
        z_hat, z_adj_hat = model.decoder(z, adj)
        a_hat = a + z_adj_hat
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, x))
        loss_a = F.mse_loss(a_hat, adj.to_dense())
        loss = loss_w + opt.args.alpha_value * loss_a
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def fmt_param(x):
    """
    Convert parameter value to a safe filename string.
    Example:
        0.03 -> 0p03
        3e-3 -> 0p003
    """
    if isinstance(x, float):
        s = f"{x:g}"
    else:
        s = str(x)
    return s.replace(".", "p").replace("-", "m")


def get_exp_name():
    trial_part = ""
    if hasattr(opt.args, "trial_id") and opt.args.trial_id >= 0:
        trial_part = f"_trial{opt.args.trial_id}"

    return (
        f"{opt.args.name}"
        f"{trial_part}"
        f"_seed{opt.args.seed}"
        f"_k{opt.args.k}"
        f"_topk{opt.args.cspm_topk}"
        f"_beta{fmt_param(opt.args.cspm_beta)}"
        f"_lp{fmt_param(opt.args.lambda_pair)}"
        f"_l2{fmt_param(opt.args.lambda2)}"
        f"_lr{fmt_param(opt.args.lr)}"
    )


def get_pretrain_path():
    save_dir = Path("./model_pretrained") / opt.args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / f"{get_exp_name()}_pretrain.pkl"


def get_output_dir():
    save_dir = Path("./output") / opt.args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_result_dir():
    save_dir = Path("./tuning_results") / opt.args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def save_result_csv(best_epoch, best_ari, best_nmi, best_ami, best_acc):
    result_dir = get_result_dir()
    csv_path = result_dir / "all_tuning_results.csv"
    json_path = result_dir / f"{get_exp_name()}_result.json"

    row = {
        "exp_name": get_exp_name(),
        "dataset": opt.args.name,
        "trial_id": opt.args.trial_id if hasattr(opt.args, "trial_id") else -1,
        "seed": opt.args.seed,
        "k": opt.args.k,
        "lr": opt.args.lr,
        "rec_epoch": opt.args.rec_epoch,
        "fus_epoch": opt.args.fus_epoch,
        "epoch": opt.args.epoch,
        "lambda1": opt.args.lambda1,
        "lambda2": opt.args.lambda2,
        "lambda3": opt.args.lambda3,
        "lambda_pair": opt.args.lambda_pair,
        "cspm_topk": opt.args.cspm_topk,
        "cspm_beta": opt.args.cspm_beta,
        "cspm_dropout": opt.args.cspm_dropout,
        "best_epoch": best_epoch,
        "ARI": best_ari,
        "NMI": best_nmi,
        "AMI": best_ami,
        "ACC": best_acc,
    }

    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=4, ensure_ascii=False)

    print(f"Saved result csv : {csv_path}")
    print(f"Saved result json: {json_path}")

def binary_cross_entropy(x_pred, x):
    #mask = torch.sign(x)
    return - torch.sum(x * torch.log(x_pred + 1e-8) + (1 - x) * torch.log(1 - x_pred + 1e-8), dim=1)

def pre_train(model, X1, A1, X2, A2, adaptive_weight, p_sample):
    print("Pretraining fusion model...")
    optimizer = Adam(model.parameters(), lr=opt.args.lr)
    for epoch in tqdm.tqdm(range(opt.args.fus_epoch)):
        # input & output
        X_hat1, Z_hat1, A_hat1, X_hat2, Z_hat2, A_hat2, _, _, zg, cl_loss,normalized_x_zinb,Final_x_ber= model(X1, A1, X2, A2,
                                                                                     adaptive_weight=adaptive_weight,
                                                                                     p_sample=p_sample, pretrain=True)
        L_REC1 = reconstruction_loss(X1, A1, X_hat1, Z_hat1, A_hat1)
        L_REC2 = reconstruction_loss(X2, A2, X_hat2, Z_hat2, A_hat2)
        loss = L_REC1 + L_REC2 + opt.args.lambda2*cl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # torch.save(model.state_dict(), './model_pretrained/{}_cell_topk{}_beta{}_pretrain.pkl'.format(opt.args.name,opt.args.cspm_topk,opt.args.cspm_beta))
    pretrain_path = get_pretrain_path()
    torch.save(model.state_dict(), pretrain_path)
    print(f"Saved pretrained model to: {pretrain_path}")

def train(model, X1, A1, X2, A2, y, p_sample, adaptive_weight):
    opt.args.ari = 0
    opt.args.nmi = 0
    opt.args.ami = 0
    opt.args.acc = 0
    best_epoch = -1
    if not opt.args.pretrain:
        # loading pretrained model
        # model.load_state_dict(
        #     torch.load('./model_pretrained/{}_cell_topk{}_beta{}_pretrain.pkl'.format(opt.args.name,opt.args.cspm_topk,opt.args.cspm_beta), map_location='cpu'))
        pretrain_path = get_pretrain_path()
        print(f"Loading pretrained model from: {pretrain_path}")

        model.load_state_dict(
            torch.load(pretrain_path, map_location='cpu')
        )

        with torch.no_grad():
            _, _, _, _, _, _, Z1, Z2, _, _,_,_ = model(X1, A1, X2, A2, p_sample=p_sample,
                                                      adaptive_weight=adaptive_weight)
        _, _, _, _, centers1 = clustering(Z1, y)
        _, _, _, _, centers2 = clustering(Z2, y)

        # initialize cluster centers
        model.cluster_centers1.data = torch.tensor(centers1).to(opt.args.device)
        model.cluster_centers2.data = torch.tensor(centers2).to(opt.args.device)
    print("Training...")

    optimizer = Adam(model.parameters(), lr=(opt.args.lr))

    pbar = tqdm.tqdm(range(opt.args.epoch), ncols=200)
    for epoch in pbar:
        # input & output
        X_hat1, Z_hat1, A_hat1, X_hat2, Z_hat2, A_hat2, Z1, Z2, zg, cl_loss,normalized_x_zinb,Final_x_ber = model(X1, A1, X2, A2,
                                                                                       p_sample=p_sample,
                                                                                       adaptive_weight=adaptive_weight)
        #CSPM改进 查看自适应参数变化
        if epoch % 50 == 0:
            beta1 = model.attlayer1.last_beta.view(-1)
            beta2 = model.attlayer2.last_beta.view(-1)

            q1 = torch.quantile(beta1, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=beta1.device))
            q2 = torch.quantile(beta2, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=beta2.device))

            print(
                "RNA beta mean {:.4f}, std {:.4f}, q10 {:.4f}, q25 {:.4f}, q50 {:.4f}, q75 {:.4f}, q90 {:.4f}".format(
                    beta1.mean().item(), beta1.std().item(),
                    q1[0].item(), q1[1].item(), q1[2].item(), q1[3].item(), q1[4].item()
                )
            )
            print(
                "ATAC beta mean {:.4f}, std {:.4f}, q10 {:.4f}, q25 {:.4f}, q50 {:.4f}, q75 {:.4f}, q90 {:.4f}".format(
                    beta2.mean().item(), beta2.std().item(),
                    q2[0].item(), q2[1].item(), q2[2].item(), q2[3].item(), q2[4].item()
                )
            )
            if epoch % 50 == 0:
                if hasattr(model, "last_loss_iic"):
                    print("loss_iic:", model.last_loss_iic.item())
                if hasattr(model, "last_loss_pair"):
                    print("loss_pair:", model.last_loss_pair.item())

        # loss_zinb = F.mse_loss(normalized_x_zinb,X1)
        # loss_Ber = torch.mean(binary_cross_entropy(Final_x_ber, X2))

        # # L_DRR = drr_loss(cons)
        # L_REC1 = reconstruction_loss(X1, A1, X_hat1, Z_hat1, A_hat1)
        # L_REC2 = reconstruction_loss(X2, A2, X_hat2, Z_hat2, A_hat2)
        # loss = loss_zinb+0.0001*loss_Ber +25*L_REC1 +25* L_REC2 +0.01*cl_loss

        loss_zinb = F.mse_loss(normalized_x_zinb, X1)
        loss_Ber = torch.mean(binary_cross_entropy(Final_x_ber, X2))

        L_REC1 = reconstruction_loss(X1, A1, X_hat1, Z_hat1, A_hat1)
        L_REC2 = reconstruction_loss(X2, A2, X_hat2, Z_hat2, A_hat2)

        # loss = loss_zinb \
        #     + 0.0001 * loss_Ber \
        #     + 25 * L_REC1 \
        #     + 25 * L_REC2 \
        #     + 0.01 * cl_loss 

        loss = (
            loss_zinb
            + opt.args.lambda3 * loss_Ber
            + opt.args.lambda1 * (L_REC1 + L_REC2)
            + opt.args.lambda2 * cl_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        kmeans = KMeans(opt.args.n_clusters, n_init=20, random_state=opt.args.seed)
        y_pre = kmeans.fit_predict(zg.data.cpu().numpy())
        # clustering & evaluation
        ari, nmi, ami, acc, y_pred = assignment(y_pre, y)

        pbar.set_postfix({'loss': '{0:1.4f}'.format(loss), 'ARI': '{0:1.4f}'.format(ari), 'NMI': '{0:1.4f}'.format(nmi),
                          'AMI': '{0:1.4f}'.format(ami), 'ACC': '{0:1.4f}'.format(acc)})

        if ari > opt.args.ari:
            opt.args.acc = acc
            opt.args.nmi = nmi
            opt.args.ari = ari
            opt.args.ami = ami
            best_epoch = epoch
            # np.save('./output/{}/seed{}_label.npy'.format(opt.args.name, opt.args.seed), y_pred)
            # np.save('./output/{}/seed{}_z.npy'.format(opt.args.name, opt.args.seed),
            #         (zg).cpu().detach().numpy())
            output_dir = get_output_dir()
            exp_name = get_exp_name()

            np.save(output_dir / f"{exp_name}_label.npy", y_pred)
            np.save(output_dir / f"{exp_name}_z.npy", (zg).cpu().detach().numpy())

        if epoch % 50 == 0 and epoch > 0:
            print(
                "\nCurrent Best -> Epoch: {}, ARI: {:.4f}, NMI: {:.4f}, AMI: {:.4f}, ACC: {:.4f}".format(
                    best_epoch,
                    opt.args.ari,
                    opt.args.nmi,
                    opt.args.ami,
                    opt.args.acc
                )
            )
    pbar.close()

    print("Best_epoch: {},".format(best_epoch), "ARI: {:.4f},".format(opt.args.ari),
          "NMI: {:.4f},".format(opt.args.nmi),
          "AMI: {:.4f}".format(opt.args.ami), "ACC: {:.4f}".format(opt.args.acc))
    save_result_csv(
        best_epoch=best_epoch,
        best_ari=opt.args.ari,
        best_nmi=opt.args.nmi,
        best_ami=opt.args.ami,
        best_acc=opt.args.acc
    )



if __name__ == '__main__':

    if opt.args.name == 'PBMC-10k':
        opt.args.lr = 3.5e-3
        opt.args.seed = 8
    elif opt.args.name == 'Ma-2020-1':
        opt.args.lr = 4e-3
        opt.args.seed = 0
    elif opt.args.name == 'GSE':
        opt.args.lr = 1e-4
        opt.args.seed = 0

    # setup
    print("setting:")

    setup_seed(opt.args.seed)

    opt.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("------------------------------")
    print("dataset       : {}".format(opt.args.name))
    print("device        : {}".format(opt.args.device))
    print("random seed   : {}".format(opt.args.seed))
    print("lambda1 value : {}".format(opt.args.lambda1))
    print("lambda2 value : {}".format(opt.args.lambda2))
    print("lambda3 value : {}".format(opt.args.lambda3))
    print("alpha value   : {:.0e}".format(opt.args.alpha_value))
    print("k value       : {}".format(opt.args.k))
    print("learning rate : {:.0e}".format(opt.args.lr))
    print("CSPM top-k    : {}".format(opt.args.cspm_topk))
    print("CSPM beta     : {:.2f}".format(opt.args.cspm_beta))
    print("CSPM dropout  : {:.2f}".format(opt.args.cspm_dropout))
    print("lambda_pair  : {}".format(opt.args.lambda_pair))
    print("trial id      : {}".format(opt.args.trial_id))
    print("exp name      : {}".format(get_exp_name()))
    print("------------------------------")

    # load data
    Xr, y, Ar = load_data(opt.args.name, 'RNA', opt.args.method, opt.args.k, show_details=False)
    Xa, y, Aa = load_data(opt.args.name, 'ATAC', opt.args.method, opt.args.k, show_details=False)
    opt.args.n_clusters = int(max(y) - min(y) + 1)
    opt.args.n_d1 = Xr.shape[1]
    # opt.args.n_d1 = Xa.shape[1]
    opt.args.n_d2 = Xa.shape[1]

    Xr = numpy_to_torch(Xr).to(opt.args.device)
    Ar = numpy_to_torch(Ar, sparse=True).to(opt.args.device)

    Xa = numpy_to_torch(Xa).to(opt.args.device)
    Aa = numpy_to_torch(Aa, sparse=True).to(opt.args.device)

    ae1 = AE(
        ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2,
        ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2,
        n_input=opt.args.n_d1, n_z=opt.args.n_z).to(opt.args.device)

    ae2 = AE(
        ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2,
        ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2,
        n_input=opt.args.n_d2, n_z=opt.args.n_z).to(opt.args.device)

    if opt.args.pretrain:
        opt.args.dropout = 0.4
    gae1 = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
        n_input=opt.args.n_d1, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

    gae2 = IGAE(
        gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
        gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
        n_input=opt.args.n_d2, n_z=opt.args.n_z, dropout=opt.args.dropout).to(opt.args.device)

    view = 3
    # init p distribution
    p_sample = np.ones(view)
    weight_history = []
    p_sample = p_sample / sum(p_sample)
    p_sample = torch.FloatTensor(p_sample).cuda()

    # init adaptive weight
    adaptive_weight = np.ones(view)
    adaptive_weight = adaptive_weight / sum(adaptive_weight)
    adaptive_weight = torch.FloatTensor(adaptive_weight).cuda()
    adaptive_weight = adaptive_weight.unsqueeze(1)

    if opt.args.pretrain:
        t0 = time()
        pretrain_ae(ae1, Xr)
        pretrain_ae(ae2, Xa)

        pretrain_gae(gae1, Xr, Ar)
        pretrain_gae(gae2, Xa, Aa)

        model = scRePair(ae1, ae2, gae1, gae2, n_node=Xr.shape[0]).to(opt.args.device)

        pre_train(model, Xr, Ar, Xa, Aa, p_sample=p_sample, adaptive_weight=adaptive_weight)
        t1 = time()
        print("Time_cost: {}".format(t1 - t0))
    else:
        t0 = time()
        model = scRePair(ae1, ae2, gae1, gae2, n_node=Xr.shape[0]).to(opt.args.device)

        train(model, Xr, Ar, Xa, Aa, y, p_sample=p_sample, adaptive_weight=adaptive_weight)
        t1 = time()
        print("Time_cost: {}".format(t1 - t0))



