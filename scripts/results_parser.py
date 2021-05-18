# -----------------------------------------------------------------------------
# Authors: 
#   - Santiago Donaher
#   - Alessio Xompero: a.xompero@qmul.ac.uk
#
# MIT License

# Copyright (c) 2021 CORSMAL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------

import pandas as pd

def estimations2csv(y_pred, container, sequence, action_pred):

    cont_cap = []
    fill_lev = []
    fill_typ = []

    for n in range(len(y_pred)):

        cont_cap.append(-1)

        if y_pred[n] == 0:
            fill_lev.append(0)
            fill_typ.append(0)
        elif y_pred[n] == 1:
            fill_lev.append(1)
            fill_typ.append(1)
        elif y_pred[n] == 2:
            fill_lev.append(2)
            fill_typ.append(1)
        elif y_pred[n] == 3:
            fill_lev.append(1)
            fill_typ.append(2)
        elif y_pred[n] == 4:
            fill_lev.append(2)
            fill_typ.append(2)
        elif y_pred[n] == 5:
            fill_lev.append(1)
            fill_typ.append(3)
        elif y_pred[n] == 6:
            fill_lev.append(2)
            fill_typ.append(3)
        else:
            fill_lev.append(-1)
            fill_typ.append(-1)

    df_results = pd.DataFrame(list(zip(container, sequence, cont_cap, fill_lev, fill_typ, action_pred)),
            columns =['Container ID', 'Sequence', 'Container Capacity', 'Filling level', 'Filling type', 'action_pred'])

    return df_results