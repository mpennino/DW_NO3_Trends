# in terminal:
## venv\Scripts\Activate

# NO3 Trends PyTorch
import pandas as pd

# Load Dataset
# DATA = readRDS(paste0(strap_dir,'Data/Models/RF_bi_model_All_DATA_all_vars_','Trends_Conc_PWS_GW_05to20', '.rds'))
strap_dir = 'C:/Users/MPennino/OneDrive - Environmental Protection Agency (EPA)/Projects/StRAPs/StRAP4/SSWR.405.1_NO3_Trend_Causes/Data/Models/'
FILE_NAME = 'RF_bi_model_All_DATA_all_vars_Trends_Conc_PWS_GW_05to20.csv'
df = pd.read_csv(strap_dir+FILE_NAME)
df.head(3)