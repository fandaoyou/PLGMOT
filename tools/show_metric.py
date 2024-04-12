import os


def print_row(begin='', items=None):
    to_print = '%-40s' % begin
    for each in items:
        to_print += '%-10s' % str(each)
    print(to_print)
    # print('-' * 120)


exps = os.listdir('./track_results')
# valid_metrics = ['MOTA', 'IDF1', 'CLR_FP', 'CLR_FN', 'IDSW']
# valid_metrics = ['MOTA', 'IDF1', 'MT', 'ML', 'IDSW']
valid_metrics = ['MOTA', 'IDF1', 'MT', 'ML', 'CLR_FP', 'CLR_FN', 'IDSW', 'CLR_Re', 'CLR_Pr']

exps.sort()
exps.sort(key=lambda x: float(x[-2:]))
print_row(items=valid_metrics)
for i in range(len(exps)):
    exp = exps[i]
    # if 'dataset' not in exp: continue
    if (i+1) % 10 == 0:
        print_row(items=valid_metrics)

    with open(f'./track_results/{exp}/pedestrian_summary.txt', 'r') as f:
        lines = f.readlines()
    metric_names, metrics = lines
    seq_result = {}
    for metric_name, metric in zip(metric_names.split(), metrics.split()):
        value = float(metric)
        if metric_name in valid_metrics:
            if metric_name in ['MOTA', 'IDF1', 'CLR_Re', 'CLR_Pr']:
                metric = f'{value:.2f}'
                metric += '%'
            seq_result[metric_name] = metric
    items = []
    for valid_metric in valid_metrics:
        items.append(seq_result[valid_metric])
    print_row(exp, items)
