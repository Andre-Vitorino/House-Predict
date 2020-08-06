import pandas as pd 
import inflection 
import math 
import datetime 
import pickle
import numpy as np


class House(object):
    def __init__(self):
        self.year_built_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/year_built_scaler.pkl', 'rb'))

        self.ms_sub_class_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/ms_sub_class_scaler.pkl', 'rb'))

        self.lot_frontage_scaler =  pickle.load(open('/home/andre/repos/House-Prices/parameter/lot_frontage_scaler.pkl', 'rb'))

        self.lot_area_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/lot_area_scaler.pkl', 'rb'))

        self.overall_cond_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/overall_cond_scaler.pkl', 'rb'))

        self.year_remod_add_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/year_remod_add_scaler.pkl', 'rb'))

        self.mas_vnr_area_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/mas_vnr_area_scaler.pkl', 'rb'))

        self.bsmt_fin_sf1_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/bsmt_fin_sf1_scaler.pkl', 'rb'))

        self.bsmt_fin_sf2_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/bsmt_fin_sf2_scaler.pkl', 'rb'))

        self.bsmt_unf_sf_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/bsmt_unf_sf_scaler.pkl', 'rb'))

        self.total_bsmt_sf_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/total_bsmt_sf_scaler.pkl', 'rb'))

        self.st_flr_sf_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/1st_flr_sf_scaler.pkl', 'rb'))

        self.nd_flr_sf_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/2nd_flr_sf_scaler.pkl', 'rb'))

        self.low_qual_fin_sf_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/low_qual_fin_sf_scaler.pkl', 'rb'))

        self.gr_liv_area_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/gr_liv_area_scaler.pkl', 'rb'))

        self.bsmt_full_bath_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/bsmt_full_bath_scaler.pkl', 'rb'))

        self.bsmt_half_bath_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/bsmt_half_bath_scaler.pkl', 'rb'))

        self.full_bath_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/full_bath_scaler.pkl', 'rb'))

        self.half_bath_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/half_bath_scaler.pkl', 'rb'))

        self.bsmt_full_bath_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/bsmt_full_bath_scaler.pkl', 'rb'))

        self.bedroom_abv_gr_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/bedroom_abv_gr_scaler.pkl', 'rb'))

        self.kitchen_qual_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/kitchen_qual_scaler.pkl', 'rb'))
        
        self.kitchen_abv_gr_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/kitchen_abv_gr_scaler.pkl', 'rb'))

        self.tot_rms_abv_grd_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/tot_rms_abv_grd_scaler.pkl', 'rb'))

        self.fireplaces_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/fireplaces_scaler.pkl', 'rb'))

        self.garage_yr_blt_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/garage_yr_blt_scaler.pkl', 'rb'))

        self.garage_cars_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/garage_cars_scaler.pkl', 'rb'))

        self.garage_area_scaler =  pickle.load(open('/home/andre/repos/House-Prices/parameter/garage_area_scaler.pkl', 'rb'))

        self.wood_deck_sf_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/wood_deck_sf_scaler.pkl', 'rb'))

        self.open_porch_sf_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/open_porch_sf_scaler.pkl', 'rb'))

        self.enclosed_porch_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/enclosed_porch_scaler.pkl', 'rb'))

        self.ssn_porch_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/3_ssn_porch_scaler.pkl', 'rb'))

        self.screen_porch_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/screen_porch_scaler.pkl', 'rb'))

        self.pool_area_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/pool_area_scaler.pkl', 'rb'))

        self.misc_val_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/misc_val_scaler.pkl', 'rb'))
        
        self.ms_zoning_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/ms_zoning_scaler.pkl', 'rb'))

        self.street_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/street_scaler.pkl', 'rb'))

        self.lot_shape_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/lot_shape_scaler.pkl', 'rb'))

        self.land_contour_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/land_contour_scaler.pkl', 'rb'))

        self.utilities_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/utilities_scaler.pkl', 'rb'))

        self.lot_config_scaler =  pickle.load(open('/home/andre/repos/House-Prices/parameter/lot_config_scaler.pkl', 'rb'))

        self.land_slope_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/land_slope_scaler.pkl', 'rb'))

        self.neighborhood_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/neighborhood_scaler.pkl', 'rb'))

        self.condition1_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/condition1_scaler.pkl', 'rb'))

        self.condition2_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/condition2_scaler.pkl', 'rb'))

        self.bldg_type_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/bldg_type_scaler.pkl', 'rb'))

        self.house_style_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/house_style_scaler.pkl', 'rb'))

        self.overall_qual_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/overall_qual_scaler.pkl', 'rb'))

        self.roof_style_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/roof_style_scaler.pkl', 'rb'))

        self.roof_matl_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/roof_matl_scaler.pkl', 'rb'))

        self.exterior1st_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/exterior1st_scaler.pkl', 'rb'))

        self.exterior2nd_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/exterior2nd_scaler.pkl', 'rb'))

        #self.mas_vnr_type_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/mas_vnr_type_scaler.pkl', 'rb'))

        self.exter_qual_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/exter_qual_scaler.pkl', 'rb'))

        self.exter_cond_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/exter_cond_scaler.pkl', 'rb'))

        self.foundation_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/foundation_scaler.pkl', 'rb'))

        self.bsmt_qual_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/bsmt_qual_scaler.pkl', 'rb'))

        self.bsmt_cond_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/bsmt_cond_scaler.pkl', 'rb'))

        self.bsmt_exposure_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/bsmt_exposure_scaler.pkl', 'rb'))

        self.bsmt_fin_type1_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/bsmt_fin_type1_scaler.pkl', 'rb'))

        self.bsmt_fin_type2_scaler =  pickle.load(open('/home/andre/repos/House-Prices/parameter/bsmt_fin_type2_scaler.pkl', 'rb'))

        self.heating_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/heating_scaler.pkl', 'rb'))

        self.heating_qc_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/heating_qc_scaler.pkl', 'rb'))

        self.central_air_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/central_air_scaler.pkl', 'rb'))

        self.electrical_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/electrical_scaler.pkl', 'rb'))

        self.functional_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/functional_scaler.pkl', 'rb'))

        self.garage_type_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/garage_type_scaler.pkl', 'rb'))

        self.garage_finish_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/garage_finish_scaler.pkl', 'rb'))

        self.garage_qual_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/garage_qual_scaler.pkl', 'rb'))

        self.garage_cond_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/garage_cond_scaler.pkl', 'rb'))

        self.paved_drive_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/paved_drive_scaler.pkl', 'rb'))

        self.sale_type_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/sale_type_scaler.pkl', 'rb'))

        self.sale_condition_scaler = pickle.load(open('/home/andre/repos/House-Prices/parameter/sale_condition_scaler.pkl', 'rb'))

       

    
    def data_cleaning(self, df1):

        columns_old = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street','Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType','HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType','MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1','BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual','TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual','GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC','Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition']


        #alterando o formato dos nomes das colunas 
        snakecase = lambda x: inflection.underscore(x)
        cols_new = list(map(snakecase,columns_old))
        df1.columns = cols_new


        # dropando as colunas que não tem como arrumar 
        df1 = df1.drop(['alley', 'pool_qc', 'fence', 'misc_feature', 'fireplace_qu'], axis=1)

        #lot_frontage 
        df1['lot_frontage'].fillna(60, inplace=True)

        #bsmt_qual            
        df1['bsmt_qual'].fillna('TA', inplace=True)

        #bsmt_cond  
        df1['bsmt_cond'].fillna('TA', inplace=True)

        #bsmt_exposure        
        df1['bsmt_exposure'].fillna('NO', inplace=True)

        #bsmt_fin_type1 
        df1['bsmt_fin_type1'].fillna('Unf', inplace=True)

        #bsmt_fin_type2
        df1['bsmt_fin_type2'].fillna('Unf', inplace=True)

        #mas_vnr_type   
        #df1['mas_vnr_type'].fillna('None', inplace=True)
        df1.drop('mas_vnr_type', axis=1)

        #mas_vnr_area 
        df1['mas_vnr_area'].fillna(0, inplace=True)

        #electrical
        df1['electrical'].fillna('SBrkr', inplace=True)

        #garage_type   
        df1['garage_type'].fillna('Attchd', inplace=True)

        #garage_yr_blt(CONVERTER PARA YEAR)   
        df1['garage_yr_blt'].fillna(2005, inplace=True)

        #garage_finish
        df1['garage_finish'].fillna('Unf', inplace=True)

        #garage_qual  
        df1['garage_qual'].fillna('TA', inplace=True)

        #garage_cond
        df1['garage_cond'].fillna('TA', inplace=True)

        # Arrumando os formatos das colunas, conforme o necessário 
        #alterando o tipo da colunas abaixo de float para int

        # Os valores de anos serão, por padrão, INT

        #garage_yr_blt
        df1['garage_yr_blt'] = df1['garage_yr_blt'].astype(int)

        #mas_vnr_area
        df1['mas_vnr_area'] = df1['mas_vnr_area'].astype(int)

        #lot_frontage
        df1['lot_frontage'] = df1['lot_frontage'].astype(int)


        #alterando as colunas com valores numéricos para o tipo INT(JÁ FEITO)

        # alterando o tipo da coluna de INT para STR
        df1['overall_qual'] = df1['overall_qual'].astype(str)
        

        
        return df1



    def feature_engineer(self, df2):
        # Fazendo a criação de novas variáveis que sejam úteis para o modelo 

        df2['sale_date'] = df2['mo_sold'].astype(str) + df2['yr_sold'].astype(str)
        df2['sale_date'] = pd.to_datetime(df2['sale_date'], format='%m%Y')
        df2 = df2.drop_duplicates()

        # Selecionando colunas para excluir
        df2 = df2.drop(['mo_sold', 'yr_sold'], axis=1)

        return df2
    
    
    
    def data_preparation(self,df1, df5):


        df5['year_built'] = self.year_built_scaler.fit_transform(df5[['year_built']].values)

        df5['ms_sub_class'] = self.ms_sub_class_scaler.fit_transform(df5[['ms_sub_class']].values)

        df5['lot_frontage'] = self.lot_frontage_scaler.fit_transform(df5[['lot_frontage']].values)

        df5['lot_area'] = self.lot_area_scaler.fit_transform(df5[['lot_area']].values)

        df5['overall_cond'] = self.overall_cond_scaler.fit_transform(df5[['overall_cond']].values)

        df5['year_remod_add'] = self.year_remod_add_scaler.fit_transform(df5[['year_remod_add']].values)

        df5['mas_vnr_area'] = self.mas_vnr_area_scaler.fit_transform(df5[['mas_vnr_area']].values)

        df5['bsmt_fin_sf1'] = self.bsmt_fin_sf1_scaler.fit_transform(df5[['bsmt_fin_sf1']].values)

        df5['bsmt_fin_sf2'] = self.bsmt_fin_sf2_scaler.fit_transform(df5[['bsmt_fin_sf2']].values)

        df5['bsmt_unf_sf'] = self.bsmt_unf_sf_scaler.fit_transform(df5[['bsmt_unf_sf']].values)

        df5['total_bsmt_sf'] = self.total_bsmt_sf_scaler.fit_transform(df5[['total_bsmt_sf']].values)

        df5['1st_flr_sf'] = self.st_flr_sf_scaler.fit_transform(df5[['1st_flr_sf']].values)

        df5['2nd_flr_sf'] = self.nd_flr_sf_scaler.fit_transform(df5[['2nd_flr_sf']].values)

        df5['low_qual_fin_sf'] = self.low_qual_fin_sf_scaler.fit_transform(df5[['low_qual_fin_sf']].values)

        df5['gr_liv_area'] = self.gr_liv_area_scaler.fit_transform(df5[['gr_liv_area']].values)

        df5['bsmt_full_bath'] = self.bsmt_full_bath_scaler.fit_transform(df5[['bsmt_full_bath']].values)

        df5['bsmt_half_bath'] = self.bsmt_half_bath_scaler.fit_transform(df5[['bsmt_half_bath']].values)

        df5['full_bath'] = self.full_bath_scaler.fit_transform(df5[['full_bath']].values)

        df5['half_bath'] = self.half_bath_scaler.fit_transform(df5[['half_bath']].values)

        df5['bedroom_abv_gr'] = self.bedroom_abv_gr_scaler.fit_transform(df5[['bedroom_abv_gr']].values)

        df5['kitchen_abv_gr'] = self.kitchen_abv_gr_scaler.fit_transform(df5[['kitchen_abv_gr']].values)

        df5['tot_rms_abv_grd'] = self.tot_rms_abv_grd_scaler.fit_transform(df5[['tot_rms_abv_grd']].values)

        df5['fireplaces'] = self.fireplaces_scaler.fit_transform(df5[['fireplaces']].values)

        df5['garage_yr_blt'] = self.garage_yr_blt_scaler.fit_transform(df5[['garage_yr_blt']].values)

        df5['garage_cars'] = self.garage_cars_scaler.fit_transform(df5[['garage_cars']].values)

        df5['garage_area'] = self.garage_area_scaler.fit_transform(df5[['garage_area']].values)

        df5['wood_deck_sf'] = self.wood_deck_sf_scaler.fit_transform(df5[['wood_deck_sf']].values)

        df5['open_porch_sf'] = self.open_porch_sf_scaler.fit_transform(df5[['open_porch_sf']].values)

        df5['enclosed_porch'] = self.enclosed_porch_scaler.fit_transform(df5[['enclosed_porch']].values)

        df5['3_ssn_porch'] = self.ssn_porch_scaler.fit_transform(df5[['3_ssn_porch']].values)

        df5['screen_porch'] = self.screen_porch_scaler.fit_transform(df5[['screen_porch']].values)

        df5['pool_area'] = self.pool_area_scaler.fit_transform(df5[['pool_area']].values)

        df5['misc_val'] = self.misc_val_scaler.fit_transform(df5[['misc_val']].values)


        #Label Encoder

        df5['ms_zoning'] = self.ms_zoning_scaler.fit_transform(df5['ms_zoning'])

        df5['street'] = self.street_scaler.fit_transform(df5['street'])

        df5['lot_shape'] = self.lot_shape_scaler.fit_transform(df5['lot_shape'])

        df5['land_contour'] = self.land_contour_scaler.fit_transform(df5['land_contour'])

        df5['utilities'] = self.utilities_scaler.fit_transform(df5['utilities'])

        df5['lot_config'] = self.lot_config_scaler.fit_transform(df5['lot_config'])

        df5['land_slope'] = self.land_slope_scaler.fit_transform(df5['land_slope'])

        df5['neighborhood'] = self.neighborhood_scaler.fit_transform(df5['neighborhood'])

        df5['condition1'] = self.condition1_scaler.fit_transform(df5['condition1'])

        df5['condition2'] = self.condition2_scaler.fit_transform(df5['condition2'])

        df5['bldg_type'] = self.bldg_type_scaler.fit_transform(df5['bldg_type'])

        df5['house_style'] = self.house_style_scaler.fit_transform(df5['house_style'])

        df5['overall_qual'] = self.overall_qual_scaler.fit_transform(df5['overall_qual'])

        df5['roof_style'] = self.roof_style_scaler.fit_transform(df5['roof_style'])

        df5['roof_matl'] = self.roof_matl_scaler.fit_transform(df5['roof_matl'])

        df5['exterior1st'] = self.exterior1st_scaler.fit_transform(df5['exterior1st'])

        df5['exterior2nd'] = self.exterior2nd_scaler.fit_transform(df5['exterior2nd'])

        #df5['mas_vnr_type'] = self.mas_vnr_type_scaler.fit_transform(df5['mas_vnr_type'])

        df5['exter_qual'] = self.exter_qual_scaler.fit_transform(df5['exter_qual'])

        df5['exter_cond'] = self.exter_cond_scaler.fit_transform(df5['exter_cond'])

        df5['foundation'] = self.foundation_scaler.fit_transform(df5['foundation'])

        df5['bsmt_qual'] = self.bsmt_qual_scaler.fit_transform(df5['bsmt_qual'])

        df5['bsmt_cond'] = self.bsmt_cond_scaler.fit_transform(df5['bsmt_cond'])

        df5['bsmt_exposure'] = self.bsmt_exposure_scaler.fit_transform(df5['bsmt_exposure'])

        df5['bsmt_fin_type1'] = self.bsmt_fin_type1_scaler.fit_transform(df5['bsmt_fin_type1'])

        df5['bsmt_fin_type2'] = self.bsmt_fin_type2_scaler.fit_transform(df5['bsmt_fin_type2'])

        df5['heating'] = self.heating_scaler.fit_transform(df5['heating'])

        df5['heating_qc'] = self.heating_qc_scaler.fit_transform(df5['heating_qc'])

        df5['central_air'] = self.central_air_scaler.fit_transform(df5['central_air'])

        df5['electrical'] = self.electrical_scaler.fit_transform(df5['electrical'])

        df5['kitchen_qual'] = self.kitchen_qual_scaler.fit_transform(df5['kitchen_qual'])

        df5['functional'] = self.functional_scaler.fit_transform(df5['functional'])

        df5['garage_type'] = self.garage_type_scaler.fit_transform(df5['garage_type'])

        df5['garage_finish'] = self.garage_finish_scaler.fit_transform(df5['garage_finish'])

        df5['garage_qual'] = self.garage_qual_scaler.fit_transform(df5['garage_qual'])

        df5['garage_cond'] = self.garage_cond_scaler.fit_transform(df5['garage_cond'])

        df5['paved_drive'] = self.paved_drive_scaler.fit_transform(df5['paved_drive'])

        df5['sale_type'] = self.sale_type_scaler.fit_transform(df5['sale_type'])

        df5['sale_condition'] = self.sale_condition_scaler.fit_transform(df5['sale_condition'])


        #VARIÁVEIS CÍCLICAS 

        df5['sale_date_sin'] = df1['mo_sold'].apply(lambda x: np.sin(x *(2. * np.pi /30)))
        df5['sale_date_cos'] = df1['mo_sold'].apply(lambda x: np.cos(x *(2. * np.pi /30)))
        
        cols_selected = ['ms_zoning','lot_area','neighborhood','overall_qual','overall_cond','year_built','year_remod_add','exter_qual','bsmt_qual','bsmt_fin_sf1',
                        'bsmt_unf_sf','total_bsmt_sf','1st_flr_sf','2nd_flr_sf','gr_liv_area','full_bath','garage_finish','garage_cars','garage_area']
        
        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        pred = model.predict(test_data)
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient='records', date_format='iso')
