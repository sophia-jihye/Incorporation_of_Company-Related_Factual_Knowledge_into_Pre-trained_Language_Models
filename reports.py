from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def create_confusion_matrix(labels, preds, out_filepath):
    confmat = pd.crosstab(pd.Series(labels, name="true"), pd.Series(preds, name="prediction"))
    confmat['n_true'] = confmat.sum(axis=1)
    confmat.loc['n_pred'] = confmat.sum(axis=0)
    confmat.to_csv(out_filepath)
    print('Created {}'.format(out_filepath))
    return confmat

def create_classification_report(labels, preds, accuracy, out_filepath):
    report = pd.DataFrame(classification_report(labels, preds, output_dict=True)).transpose()
    report.loc['accuracy'] = accuracy
    report.to_csv(out_filepath)
    print('Created {}'.format(out_filepath))
    return report

def get_average_classification_report(reports):
    for i, report in enumerate(reports):
        report.set_index(report.columns[0], inplace=True)
        if i == 0: 
            report_avg = report
        else: 
            report_avg = report_avg.add(report)
    report_avg = report_avg / len(reports)
    return report_avg

def average_classification_report(report_filepaths):
    for i, report_filepath in enumerate(report_filepaths):
        report = pd.read_csv(report_filepath, header=0, index_col=0)
        if i == 0: report_avg = report
        else: report_avg = report_avg.add(report)
    report_avg = report_avg / len(report_filepaths)
    return report_avg