from fpdf import FPDF
from datetime import date
# https://pyfpdf.readthedocs.io/en/latest/ReferenceManual/index.html


class PDF(FPDF):
    def header(self):
        epw = self.w - 2 * self.l_margin
        col_width = epw / 4
        th = 10
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Summary of Score Fusion Experiments', 0, 1, 'C')

        self.set_font('Arial', 'B', 12)
        self.cell(col_width, th, 'Generated on:', border=0)
        self.set_font('Arial', '', 12)
        self.cell(col_width, th, str(date.today()), border=0)
        self.ln(th)

    def add_table(self, data):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')


def generate_summary(modalities=None, results=None, roc_plt=None, fmr_rate=0.01, save_to_path='./generated/'):
    pdf = PDF(format='Letter')
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font('Times', '', 12)
    epw = pdf.w - 2 * pdf.l_margin
    col_width = epw / 4
    th = 10

    rules = [x for x in results.keys() if 'Rule' in x]

    modalities = [x for x in modalities if 'Rule' not in x]
    if len(modalities) < 10:
        modalities_string = ', '.join(modalities)
    else:
        modalities_string = 'Large number of modalities detected ('+str(len)+' modalities)'

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(col_width, th, 'Fusion Rules Applied:', border=0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(col_width, th, ', '.join(rules), border=0)
    pdf.ln(th)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(col_width, th, 'Modalities:', border=0)
    pdf.set_font('Arial', '', 12)
    pdf.cell(col_width, th, modalities_string, border=0)
    pdf.ln(20)

    pdf.image(roc_plt, x=col_width/2, w=140)
    pdf.ln(th)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(col_width, th, 'Fusion Rule', border=0)
    pdf.cell(col_width, th, 'AUC', border=0)
    pdf.cell(col_width, th, 'EER', border=0)
    pdf.cell(col_width, th, 'TMR @'+str(fmr_rate)+'FMR', border=0)
    pdf.line(x1=pdf.l_margin, y1=200, x2=epw, y2=200)
    pdf.ln(th)

    for rule_key in results.index:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(col_width, th, rule_key, border=0)
        pdf.set_font('Arial', '', 12)

        pdf.cell(col_width, th, str(round(results.loc[rule_key]['AUC'], 5)), border=0)
        pdf.cell(col_width, th, str(round(results.loc[rule_key]['EER'], 5)), border=0)
        pdf.cell(col_width, th, str(round(results.loc[rule_key]['TPRS'], 5)), border=0)
        pdf.ln(th)

    pdf.output(save_to_path+'ResultsSummary-'+str(date.today())+'.pdf', 'F')

