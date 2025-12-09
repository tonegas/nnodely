import io

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

from mplplots import plots


class Reporter:
    def __init__(self, modely):
        self.modely = modely

    def exportReport(self, report_path):
        c = canvas.Canvas(report_path, pagesize=letter)
        width, height = letter

        if 'Minimizers' in self.modely._model_def:
            for key, value in self.modely._model_def['Minimizers'].items():
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(111)
                if 'val' in self.modely._training[key]:
                    plots.plot_training(ax, f"Training Loss of {key}", key, self.modely._training[key]['train'], self.modely._training[key]['val'])
                else:
                    plots.plot_training(ax, f"Training Loss of {key}", key, self.modely._training[key]['train'])
                training = io.BytesIO()
                plt.savefig(training, format='png')
                training.seek(0)
                plt.close()
                c.drawString(100, height - 30, f"Training Loss of {key}")
                c.drawImage(ImageReader(training), 50, height - 290, width=500, height=250)
                c.showPage()

            if len(self.modely.prediction) > 0:
                for key in self.modely._model_def['Minimizers'].keys():
                    c.drawString(100, height - 30, f"Prediction of {key}")
                    for ind, name_data in enumerate(self.modely.prediction.keys()):
                        fig = plt.figure(figsize=(10, 5))
                        ax = fig.add_subplot(111)
                        idxs = None
                        if 'idxs' in self.modely.prediction[name_data]:
                            idxs = self.modely.prediction[name_data]['idxs']
                        plots.plot_results(ax, name_data, key, self.modely.prediction[name_data][key]['A'],
                                       self.modely.prediction[name_data][key]['B'], idxs, self.modely._model_def['Info']["SampleTime"])
                        # Add a text box with correlation coefficient
                        results = io.BytesIO()
                        plt.savefig(results, format='png')
                        results.seek(0)
                        plt.close()
                        c.drawImage(ImageReader(results), 50, height - 290 - 245*ind, width=500, height=250)
                    c.showPage()
        else:
            c.drawString(100, height - 30, f"No Minimize")
        c.save()
