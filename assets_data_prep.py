import pandas as pd
import numpy as np
import joblib
def prepare_data(df):
    df=df.copy()
    params = joblib.load("all_params.pkl")
#מילוי ערכים בוליאנים
    boolean_modes = params['boolean_modes']
    df[boolean_modes.index] = df[boolean_modes.index].fillna(boolean_modes)

#מילוי שטח חסר לפי מס חדרים
    area_means_by_room = params['area_means_by_room']
    df.loc[df["area"].isnull(), "area"] = df.loc[df["area"].isnull(), "room_num"].map(area_means_by_room)

#מילוי ועד בית
    group_means = params['group_means']
    overall_mean = params['overall_mean']
    df = df.merge(group_means, on=["neighborhood", "elevator"], how='left', suffixes=('', '_group_mean'))
    df["building_tax"] = df["building_tax"]
    df["building_tax"] = df["building_tax"].fillna(overall_mean)


#מילוי ערכים חסרים של קומה וסך קומות
    neighborhood_avg =  params['neighborhood_avg']
    floor_map = neighborhood_avg.set_index("neighborhood")["floor"]
    total_floors_map = neighborhood_avg.set_index("neighborhood")["total_floors"]

    df["floor"] = df["floor"].fillna(df["neighborhood"].map(floor_map))
    df["total_floors"] = df["total_floors"].fillna(df["neighborhood"].map(total_floors_map))
    mask = df["total_floors"] < df["floor"]
    df.loc[mask, "total_floors"] = df.loc[mask, "floor"]

#מילוי ערכים חסרים של שטח דירה
    area_means_by_room = params['area_means_by_room']
    overall_area_mean = params['overall_area_mean']
    df["area"] = df["area"].fillna(df["room_num"].map(area_means_by_room).fillna(overall_area_mean))

#פונקציה למילוי ארנונה חודשית לפי שטח
    def bin_and_fill_arnona(df, column='area', target='monthly_arnona', q=4):
        # אם העמודה לא קיימת – צור אותה עם NaN
        if target not in df.columns:
            df[target] = np.nan

        # קבלת הפרמטרים מראש
        mean_by_range = params["mean_by_range"]
        bins = params["bins"]
        labels = range(len(bins) - 1)

        # מסכה לערכים חסרים או אפס
        missing_mask = df[target].isna() | (df[target] == 0)

        for i, row in df[missing_mask].iterrows():
            area = row[column]
            if pd.notna(area):
                range_label = pd.cut([area], bins=bins, labels=labels, include_lowest=True)[0]
                fill_value = mean_by_range.get(range_label, None)
                if pd.notna(fill_value):
                    df.at[i, target] = fill_value

        return df


    df = bin_and_fill_arnona(df)
#מילוי מרחק מהמרכז לפי ממוצעים
    neighborhood_mean = params["neighborhood_mean"]
    overall_mean = params["overall_mean"]
    df["distance_from_center"] = None
    df["distance_from_center"] = df["distance_from_center"].fillna(df["neighborhood"].map(neighborhood_mean))
    df["distance_from_center"] = df["distance_from_center"].fillna(overall_mean)


    df["room_num"] = df["room_num"].astype(float)
    df['room_per_area'] = (df['area']/df['room_num'])
    df.drop("room_num", axis=1, inplace=True)
    
#קידוד וסקיילינג
    def encode_and_scale(df, train_test,params):
        df = df.copy()
        scaler = params['scaler']
        encoder_oh = params['encoder_oh']
        encoder_target = params['encoder_target']
        numeric_cols = params['numeric_cols']

        X = df.copy()
        if 'neighborhood' in X.columns:
            X['neighborhood'] = encoder_target.transform(X[['neighborhood']])
        if 'property_type' in X.columns:
            oh_encoded = encoder_oh.transform(X[['property_type']])
            oh_df = pd.DataFrame(oh_encoded, columns=encoder_oh.get_feature_names_out(['property_type']), index=X.index)
            X = pd.concat([X.drop(columns=['property_type']), oh_df], axis=1)
        X[numeric_cols] = scaler.transform(X[numeric_cols])
        return X
    df = encode_and_scale(df, 'test', params)
    df.drop("ac", axis=1, inplace=True)
    return df