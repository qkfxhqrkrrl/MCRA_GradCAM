#%%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.RTR_Maker import RTR_Maker
import numpy as np
import pandas as pd

from glob import glob 
from tqdm import tqdm
import shutil

# plt.rcParams['font.family'] = "NanumSquare"


#%% 

class TempArgs:
    pass

if __name__ == "__main__":
    # parser.add_argument('--deg', type=float, default=0.2, help='Degree range')
    # parser.add_argument('--gap', type=float, default=0.1, help='Matrix gap')
    # parser.add_argument('--size', type=str, default="300by300", help='heat map size')
    # parser.add_argument('--postfix', type=str, default="default", help='folder name')
    # args = parser.parse_args()
    
    for dist in [5, 10, 20, 30]:
        args = TempArgs()
        args.dist = dist
        args.gap = 0.1
        args.deg = 0.2
        args.size = f"{args.dist*10}by{args.dist*10}"
        args.postfix = "default"
        
        heat_map_size = args.size
        data_name = args.postfix
        isAccident = True # True : 사고 ESD용
        n_sample = 25 # 무사고용
        deg_gap = args.deg # 목표 위경도 값에서 +- 범위 값
        mat_gap = args.gap # ESD 값 간격
        dist = args.dist
        print("dist : ", dist)
        
        par_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), * [os.pardir] * 2))
        
        result_dir = os.path.join(par_dir, "data", "prq_data", f"esd_{heat_map_size}_{data_name}")
        os.makedirs(result_dir, exist_ok=True)
        
        df_stat = pd.read_csv(os.path.join(par_dir, "data", "stat_data", "maritime_accidents_statistics_filtered.csv"))
        # print(df_stat.head())
        
        # display(df_stat.head(2)) 
        stat_case = df_stat[["timestamp", "lat", "lon", "해양사고발생(시)"]]
        stat_case["timestamp"] = pd.to_datetime(stat_case["timestamp"], format="%Y-%m-%d")
        
        parquet_paths = glob(os.path.join(par_dir, "data", "geo_data", "*.parquet"))
        print(parquet_paths[0])
        parquet_paths = {str(path.replace(os.path.dirname(path) + os.path.sep, "").replace(".parquet", "")) : path for path in parquet_paths}
        
        stat_case = stat_case.sort_values("timestamp")
        # print(stat_case.shape)
        
        one_day = pd.Timedelta(2, unit="d")
        
        file = open(os.path.join(result_dir, f"infos.txt"), "w")
        
        file.write("index,day,hour,lat,lon\n")
        loop = tqdm(stat_case.values)
        for target_day_, target_lat, target_lon, target_hour in loop:
            # break
            dir_name = "{}_{}".format(target_day_.strftime("%Y_%m_%d"), target_hour)
            for target_day in pd.date_range(target_day_-one_day, target_day_+one_day):
                try:
                    df = pd.read_parquet(parquet_paths[target_day.strftime("%Y-%m-%d")])
                except:
                    # 앞뒤로 보다보면 하루를 빼거나 더하는 경우가 불가능한 경우가 생김 그렇다고 지금 하나 하나 귀한 상황에서 버려지긴 아까우니 예외 처리하기
                    continue
                    
                # 중간에 msg_type 같은 값은 필요없지만 일단 다른 데이터에 존재해서 그냥 zeros로 땜빵 때우기
                maker = RTR_Maker(df)
                # loop.set_postfix({"Day" : target_day.strftime("%Y-%m-%d"), "Lat" : target_lat, "Lon" : target_lon, "shape" : df.shape})
                
                maker.make_rtr_from_target(
                    target_info = (target_day, target_lat, target_lon), 
                    deg_gap=deg_gap, 
                    dist=dist, 
                    loop=loop,
                    save_dir=os.path.join(result_dir, dir_name)
                )
        print("\n\nMoving to other folders...")

        data_dir = os.path.join(par_dir, "data", "prq_data")
        esd_dir = os.path.join(data_dir, f"rtr_{args.dist*10}by{args.dist*10}_{args.postfix}")
        os.makedirs(esd_dir, exist_ok=True)
        save_dir = os.path.join(data_dir, f"rtr_{args.dist*10}by{args.dist*10}_filtered")
        os.makedirs(save_dir, exist_ok=True)

        #! _______________  Accident 있는 경우만 걸러주고 case_num 재정의   ___________________

        for dir_name in tqdm(os.listdir(esd_dir)):
            if ".txt" in dir_name : continue
            # print(dir_name)
            acc_year, acc_month, acc_day, acc_hour = [int(var) for var in dir_name.split("_")]
            file_names = os.listdir(os.path.join(esd_dir, dir_name))
            if len(file_names) == 0:
                continue
            else :
                lat, lon = file_names[0].rstrip(".parquet").split("_")[-2:]
                doExist = False
                for minute in range(0, 60, 5):
                    acc_file_name = f"{acc_year}_{acc_month:02d}_{acc_day:02d}_{acc_hour}_{minute}_{lat}_{lon}.parquet"
                    if acc_file_name in file_names: doExist = True
                if not doExist : continue # 사고 당일이 없다면
                
                new_dir = os.path.join(save_dir, f"{acc_year}_{acc_month}_{acc_day}_{acc_hour}")
                os.makedirs(new_dir, exist_ok=True)
                for file_name in file_names:
                    shutil.copy(os.path.join(esd_dir, dir_name, file_name), os.path.join(new_dir, file_name))

    
    
#%%

    