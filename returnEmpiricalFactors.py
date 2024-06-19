import numpy as np
import openpyxl

def returnFactor(file,sheet,domain):

    database = openpyxl.load_workbook(file, data_only=True)[sheet]
    factor = []
    for n in np.arange(2,400):
        if database['A'+str(n)].value == 'None':
            break
        else:
            if database['C'+str(n)].value == domain:
                factor.append(database['D'+str(n)].value)

    return factor
