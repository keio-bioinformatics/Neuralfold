def get_score(ref_set, pred_set, global_average=True):
    def flattern(paren_list):
        if len(paren_list) == 0 or type(paren_list[0][0]) is int:
            return paren_list
        else:
            ret = []
            for p in paren_list:
                ret += p
            return ret

    def get_counts(ref_str, pred_str):
        ref_str = set(flattern(ref_str))
        pred_str = set(flattern(pred_str))
        tp = len(ref_str & pred_str)
        fn = len(ref_str) - tp
        fp = len(pred_str) - tp
        return tp, fp, fn

    def calc(tp, fp, fn):
        sen = tp / (tp + fn)
        ppv = tp / (tp + fp)
        f_val = 2 * (sen * ppv) / (sen + ppv) if sen + ppv > 0.0 else 0.
        return sen, ppv, f_val

    if global_average:
        g_tp, g_fn, g_fp = 0, 0, 0
        for ref_str, pred_str in zip(ref_set, pred_set):
            tp, fp, fn = get_counts(ref_str, pred_str)
            g_tp += tp
            g_fp += fp
            g_fn += fn
        return calc(g_tp, g_fp, g_fn)
    else:
        sen, ppv, f_val, n = 0., 0., 0., 0
        for ref_str, pred_str in zip(ref_set, pred_set):
            l_sen, l_ppv, l_f_val = calc(*get_counts(ref_str, pred_str))
            sen += l_sen
            ppv += l_ppv
            f_val += l_f_val
            n += 1
        return sen/n, ppv/n, f_val/n
