from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, r
pandas2ri.activate()
import pandas as pd

sk = importr("ScottKnottESD")
data = pd.DataFrame({
    "TechniqueA" : [5, 1, 4],
    "TechniqueB" : [6, 8, 3],
    "TechniqueC" : [7, 10, 15],
    "TechniqueD" : [7, 10.1, 15],
})

display(data)
r_sk = sk.sk_esd(data)
column_order = list(r_sk[3]-1)
ranking = pd.DataFrame({
    "Technique": [data.columns[1] for i in column_order],
    "Rank": r_sk[0].astype(int),
    "Group": r_sk[1].astype(int),
}) # Long Format
ranking = pd.DataFrame(
    [r_sk[1].astype(int)], columns=[data.columns[i-1] for i in column_order],
) # Wide Format
