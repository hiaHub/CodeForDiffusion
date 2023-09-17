import torch
def accu(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


                pred1_train, pred2_train = accu(outputs, lables, topk=(1, ))
                train_top1.update(pred1_train[0], val_images.size(0))
                #train_top2.update(pred2_train[0], val_images.size(0))
                t_top1 = train_top1.avg
                #t_top2 = train_top2.avg

print('[epoch %d] train_loss: %.3f  test_loss: %.3f val_accuracy: %.3f top1: %.4f' %
              (epoch + 1, running_loss / train_steps, testing_loss / test_steps , val_accurate, t_top1))

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count
