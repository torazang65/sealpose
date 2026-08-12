from __future__ import print_function, absolute_import, division

import time

import torch
from utils.loss import compute_AUC, compute_PCK, mpjpe, p_mpjpe, test_score
from utils.pck import keypoint_3d_auc, keypoint_3d_pck
from utils.utils import AverageMeter
from progress.bar import Bar


####################################################################
# ### evaluate p1 p2 pck auc dataset with test-flip-augmentation
####################################################################
def evaluate(data_loader, model_pos_eval, device, key="", flipaug="", is_h3wb=False, base_joint=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_p1 = AverageMeter()
    epoch_p2 = AverageMeter()
    epoch_pck = AverageMeter()
    epoch_auc = AverageMeter()

    # Switch to evaluate mode
    model_pos_eval.eval()
    end = time.time()

    bar = Bar("Eval posenet on {}".format(key), max=len(data_loader))

    if is_h3wb:
        target_list = []
        predict_list = []

    for i, temp in enumerate(data_loader):
        targets_3d, inputs_2d = temp[0], temp[1]

        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)
        inputs_2d = inputs_2d.to(device)

        with torch.no_grad():
            if flipaug and not is_h3wb:  # flip the 2D pose Left <-> Right
                joints_left = [4, 5, 6, 10, 11, 12]
                joints_right = [1, 2, 3, 13, 14, 15]
                out_left = [4, 5, 6, 10, 11, 12]
                out_right = [1, 2, 3, 13, 14, 15]

                inputs_2d_flip = inputs_2d.detach().clone()
                inputs_2d_flip[:, :, 0] *= -1
                inputs_2d_flip[:, joints_left + joints_right, :] = inputs_2d_flip[
                    :, joints_right + joints_left, :
                ]
                outputs_3d_flip = (
                    model_pos_eval(inputs_2d_flip.view(num_poses, -1))
                    .view(num_poses, -1, 3)
                    .cpu()
                )
                outputs_3d_flip[:, :, 0] *= -1
                outputs_3d_flip[:, out_left + out_right, :] = outputs_3d_flip[
                    :, out_right + out_left, :
                ]

                outputs_3d = (
                    model_pos_eval(inputs_2d.view(num_poses, -1))
                    .view(num_poses, -1, 3)
                    .cpu()
                )
                outputs_3d = (outputs_3d + outputs_3d_flip) / 2.0
            elif flipaug and is_h3wb:
                joints_left_body = [1, 3, 5, 7, 9, 11, 13, 15]
                joints_right_body = [2, 4, 6, 8, 10, 12, 14, 16]
                joints_left_foot = [17, 18, 19]
                joints_right_foot = [20, 21, 22]
                joints_left_hand = list(range(91, 112))
                joints_right_hand = list(range(112, 133))
                joints_left_face = (
                    list(range(32, 40)) # 턱 
                    + list(range(45, 50)) # 눈썹
                    + list(range(65, 71)) # 눈
                    + [57, 58] # 코
                    + list(range(75, 80)) # 입술
                    + [86, 87] # 입
                )
                joints_right_face = (
                    list(range(30, 22, -1))
                    + list(range(44, 39, -1))
                    + [62, 61, 60, 59, 64, 63]
                    + [55, 54]
                    + [73, 72, 71, 82, 81]
                    + [84, 83]
                )
                joints_left = (
                    joints_left_body + joints_left_foot + joints_left_hand + joints_left_face
                )
                joints_right = (
                    joints_right_body
                    + joints_right_foot
                    + joints_right_hand
                    + joints_right_face
                )
                out_left = joints_left
                out_right = joints_right
                inputs_2d_flip = inputs_2d.detach().clone()
                inputs_2d_flip[:, :, 0] *= -1
                inputs_2d_flip[:, joints_left + joints_right, :] = inputs_2d_flip[
                    :, joints_right + joints_left, :
                ]
                outputs_3d_flip = (
                    model_pos_eval(inputs_2d_flip.view(num_poses, -1))
                    .view(num_poses, -1, 3)
                    .cpu()
                )
                outputs_3d_flip[:, :, 0] *= -1
                outputs_3d_flip[:, out_left + out_right, :] = outputs_3d_flip[
                    :, out_right + out_left, :
                ]

                outputs_3d = (
                    model_pos_eval(inputs_2d.view(num_poses, -1))
                    .view(num_poses, -1, 3)
                    .cpu()
                )
                outputs_3d = (outputs_3d + outputs_3d_flip) / 2.0
                
            else:
                outputs_3d = (
                    model_pos_eval(inputs_2d.view(num_poses, -1))
                    .view(num_poses, -1, 3)
                    .cpu()
                )

        # caculate the relative position.
        targets_3d = (
            targets_3d[:, :, :] - targets_3d[:, base_joint:base_joint+1, :]
        )  # the output is relative to the 0 joint
        outputs_3d = (
            outputs_3d[:, :, :] - outputs_3d[:, base_joint:base_joint+1, :]
        )  # the output is relative to the 0 joint
        
        if base_joint == 14:
            pck = keypoint_3d_pck(
                pred=outputs_3d * 1000,
                gt=targets_3d * 1000,
                mask=None,
                threshold=150
            )
            # pck = compute_PCK(targets_3d, outputs_3d)
            auc = keypoint_3d_auc(
                pred=outputs_3d * 1000,
                gt=targets_3d * 1000,
                mask=None,
            )
            # auc = compute_AUC(targets_3d, outputs_3d)
            epoch_pck.update(pck, num_poses)
            epoch_auc.update(auc, num_poses)
            

        # compute p1 and p2
        p1score = mpjpe(outputs_3d, targets_3d).item() * 1000.0
        epoch_p1.update(p1score, num_poses)
        p2score = p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0
        epoch_p2.update(p2score, num_poses)

        # for pelvis aligned MPJPE
        if is_h3wb:
            target_list.append(targets_3d)
            predict_list.append(outputs_3d)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = (
            "({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} "
            "| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}".format(
                batch=i + 1,
                size=len(data_loader),
                data=data_time.avg,
                bt=batch_time.avg,
                ttl=bar.elapsed_td,
                eta=bar.eta_td,
                e1=epoch_p1.avg,
                e2=epoch_p2.avg,
            )
        )
        bar.next()

    if is_h3wb:
        target_list = torch.cat(target_list).reshape(-1, 133, 3)
        predict_list = torch.cat(predict_list).reshape(-1, 133, 3)
        pelvis_aligned_mpjpe = test_score(predict_list, target_list, False)
    else:
        pelvis_aligned_mpjpe = None

    bar.finish()
    
    if base_joint == 14:
        # print(f"pck: {epoch_pck.avg}, auc: {epoch_auc.avg}")
        return epoch_p1.avg, epoch_p2.avg, pelvis_aligned_mpjpe, epoch_pck.avg, epoch_auc.avg
    
    return epoch_p1.avg, epoch_p2.avg, pelvis_aligned_mpjpe


#########################################
# overall evaluation function
#########################################
def evaluate_posenet(
    args, data_dict, model_pos, model_pos_eval, device, summary, writer, tag
):
    """
    evaluate H36M and 3DHP
    test-augment-flip only used for 3DHP as it does not help on H36M.
    """
    with torch.no_grad():
        model_pos_eval.load_state_dict(model_pos.state_dict())
        h36m_p1, h36m_p2 = evaluate(
            data_dict["H36M_test"],
            model_pos_eval,
            device,
            summary,
            writer,
            key="H36M_test",
            tag=tag,
            flipaug="",
        )  # no flip aug for h36m
        dhp_p1, dhp_p2 = evaluate(
            data_dict["mpi3d_loader"],
            model_pos_eval,
            device,
            summary,
            writer,
            key="mpi3d_loader",
            tag=tag,
            flipaug="_flip",
        )
    return h36m_p1, h36m_p2, dhp_p1, dhp_p2
