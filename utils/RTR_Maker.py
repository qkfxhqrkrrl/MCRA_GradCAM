#%%
import numpy as np
import pandas as pd

import os
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import datetime
from haversine import haversine_vector, Unit
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class RTR_Maker:
    def __init__(self, ais_data) -> None:
        self.data = ais_data
        
    def calc_axis(self, o_lat, o_lon, t_lat, t_lon):
        # Absolute position
        x_dist = haversine_vector(np.vstack((o_lat, o_lon)).T, np.vstack((o_lat, t_lon)).T)
        y_dist = haversine_vector(np.vstack((o_lat, o_lon)).T, np.vstack((t_lat, o_lon)).T)
        x_sign = np.where(o_lon > t_lon, -1, 1)
        y_sign = np.where(o_lat > t_lat, -1, 1)
        x_dist = x_dist * x_sign
        y_dist = y_dist * y_sign
                
        return tuple(np.vstack((x_dist, y_dist)).T)

    def calc_new(self, axis_dist, h):
        # Relative position
        axis_dist = np.array(axis_dist)
        x = axis_dist[:,0]
        y = axis_dist[:,1]

        x_dist = (np.cos(np.radians(h)-np.pi/2)* x) + (np.sin(np.radians(h)-np.pi/2) * y)
        y_dist = -(np.sin(np.radians(h)-np.pi/2)* x) + (np.cos(np.radians(h)-np.pi/2) * y)
        
        return np.vstack((x_dist, y_dist)).T
    
    def calculate_dist(self, target_df):
        # Every combination of two ships with positional information is now dataframed 
        target_df = target_df.join(target_df[['mmsi','lat', 'lon']], rsuffix='_t')
        target_df = target_df[target_df['mmsi'] != target_df['mmsi_t']]
        
        if target_df.empty :
            return None

        target_df = target_df.rename(columns= {'mmsi':'mmsi_o'})

        # Calculate the haversine distance
        target_df['dist'] = haversine_vector(np.vstack((target_df['lat'], target_df['lon'])).T, \
            np.vstack((target_df['lat_t'], target_df['lon_t'])).T, Unit.KILOMETERS)
        if target_df.empty :
            return None
        target_df['axis_dist'] = self.calc_axis(target_df['lat'], target_df['lon'], target_df['lat_t'], target_df['lon_t'])
        target_df[['x_dist','y_dist']] = self.calc_new(target_df['axis_dist'].to_list(), target_df['course'])
        
        return target_df
    
    def make_base_matrix(self, dist, dist_gap):
        axis = np.arange(-dist, dist+dist_gap, dist_gap).round(1)
        axis[int(axis.size/2)] = 0
        matrix = pd.DataFrame(data = 0, index = axis, columns=axis)
        return matrix, axis

    def make_matrix(self, mat_data, dist):
        # Make RTR based on the distance
        matrix, axis = self.make_base_matrix(dist, dist_gap=0.1)
        mat_data["x_dist"] = np.round(mat_data["x_dist"], 1)
        mat_data["y_dist"] = np.round(mat_data["y_dist"], 1)
        mat_data = mat_data.loc[(mat_data["x_dist"] < dist) & (mat_data["x_dist"] > -dist) & \
            (mat_data["y_dist"] < dist) & (mat_data["y_dist"] > -dist)]
        for x_dist, y_dist, count in mat_data[["x_dist", "y_dist"]].groupby(["x_dist", "y_dist"], as_index=False).value_counts().values:
            matrix.loc[x_dist, y_dist] = count
        
        return matrix, axis
    
    def get_target_grid(self, gap, target_lat, target_lon):
        # Extract data that lies in target location and within certain distance (0.1 degree)
        # This is not for RTR, but to make operation more efficient 
        target_grid = self.data.loc[(self.data["lat"] >= target_lat-gap) & (self.data["lat"] < target_lat + gap) & \
            (self.data["lon"] >= target_lon-gap) & (self.data["lon"] < target_lon + gap)].copy()
        if len(target_grid) == 0:
            return None
        if target_grid["timestamp"].dtype != "datetime64[ns]":
            target_grid["timestamp"] = pd.to_datetime(target_grid["timestamp"])
        
        # Resample the data by 30 seconds for every target ships
        target_df = pd.DataFrame()
        for mmsi in target_grid["mmsi"].unique():
            df_mini = target_grid.loc[target_grid["mmsi"] == mmsi, :]
            time_range = pd.date_range(start=min(df_mini.timestamp) ,end=max(df_mini.timestamp), freq="s")
            df_interpol = pd.DataFrame(time_range, columns=["timestamp"])
            df_interpol = pd.merge(df_interpol, df_mini, how="left", on="timestamp")
            df_interpol.set_index("timestamp", drop=True, inplace=True)
            df_interpol = df_interpol.resample('30s').first()
            df_interpol["mmsi"].fillna(method="pad", inplace=True)
            df_interpol.interpolate(inplace=True)
            target_df = pd.concat([target_df, df_interpol])
        
        target_df = target_df[target_df.course< 360] # Empty course data is removed
        
        return target_df
        
    def make_rtr_from_target(self, target_info, deg_gap, dist, save_dir, loop : tqdm = None):
        os.makedirs(save_dir, exist_ok=True)
        
        target_day, target_lat, target_lon = target_info

        target_grid = self.get_target_grid(
                target_lat=target_lat,
                target_lon=target_lon,
                gap=deg_gap
            )

        if target_grid is None :
            if loop is not None:
                word = "There is no other ships in the area".ljust(45, " ")
                loop.set_description(word)
            else :
                print(word)
            return False
        else :
            word = f"Selected grid has shape of {target_grid.shape}".ljust(45, " ")
            target_grid_ = target_grid.copy()
            # print(target_grid_)
            
            minutes = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60] 
            os.makedirs(save_dir, exist_ok=True)
            for hour in range(24):
                for idx, min in enumerate(minutes[:-1]):
                    # Resample by every 5 minutes
                    target_grid = target_grid_.loc[(target_grid_.index.hour == hour) &\
                                                    (target_grid_.index.minute <= minutes[idx+1]) &\
                                                    (target_grid_.index.minute >= minutes[idx])]

                    if loop is not None:
                        loop.set_description(word)
                    else :
                        print(word)

                    target_grid = self.calculate_dist(target_grid)

                    if target_grid is None :
                        word = "There is no other ships in the area".ljust(45, " ")
                        if loop is not None:
                            loop.set_description(word)
                        else :
                            print(word)
                        return False
                    else :
                        word = f"Distance calculation is done - {target_grid.shape}".ljust(45, " ")
                        if loop is not None:
                            loop.set_description(word)
                        else :
                            print(word)

                    target_grid = pd.DataFrame(target_grid.reset_index(drop = True))
                    esd_mat, axis = self.make_matrix(target_grid, dist)
                    esd_mat.columns = esd_mat.columns.astype(str)
                    esd_mat.to_parquet(os.path.join(save_dir, f"{target_day.strftime('%Y_%m_%d')}_{hour}_{min}_{target_lat:.4f}_{target_lon:.4f}.parquet"), engine="pyarrow")

        return True
    

if __name__ == "__main__":
    print("Hello")