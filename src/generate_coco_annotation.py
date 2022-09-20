import os 
import json
from shapely.geometry import Polygon
from OSMPythonTools.api import Api
import pandas as pd
from shapely.geometry import Polygon
import copy
import cv2
import math
import shapely.geometry as geom

'''
Cretes COCO Annotation using VIA annotation

Example VIA annotation format:
{
   "FLAT.213253160.jpg43089":{
      "fileref":"",
      "size":43089,
      "filename":"FLAT.213253160.jpg",
      "base64_img_data":"",
      "file_attributes":{

      },
      "regions":{
         "0":{
            "shape_attributes":{
               "name":"polygon",
               "all_points_x":[
                  205,
                  304,
                  415,
                  324,
                  205
               ],
               "all_points_y":[
                  40,
                  394,
                  362,
                  39,
                  40
               ]
            },
            "region_attributes":{
               "building":"flat"
            }
         }
      }
   }
}, 
Example COCO annotation format: 
{
   "info":{
      "contributor":"",
      "about":"",
      "date_created":"",
      "description":"",
      "url":"",
      "version":"",
      "year":2019
   },
   "categories":[
      {
         "id":100,
         "name":"building",
         "supercategory":"building"
      }
   ],
   "images":[
      {
         "id":20289,
         "file_name":"000000020289.jpg",
         "width":300,
         "height":300,
         "lat":0,
         "lon":0,
         zoom=20
      }
   ],
   "annotations":[
      {
         "id":377545,
         "image_id":44153,
         "segmentation":[
            [
               152.0,
               180.0,
               156.0,
               176.0,
               160.0,
               181.0,
               156.0,
               186.0,
               152.0,
               180.0
            ]
         ],
         "area":42.0,
         "bbox":[
            152.0,
            152.0,
            28.0,
            8.0
         ],
         "category_id":100,
         "iscrowd":0
      }
   ]
}
'''

api = Api()
sunroof_data = "../data/sunroof_cities.csv"
df = pd.read_csv(sunroof_data)

# DO NOT CHANGE - USED To train MaskRCNN model "id:0 is background == BG"
CATEGORIES_LIST = ["flat","dome", "N", "NNE", "NE", "ENE", "E", "ESE", "SE","SSE", "S", "SSW","SW","WSW", "W", "WNW", "NW", "NNW","tree"]
CATEGORIES =[ {"id": i+1, "name": cat, "supercategory": "building" } for i, cat in enumerate(CATEGORIES_LIST) ]
CATEGORIES_MAP = dict( (cat["name"], cat["id"]) for cat in CATEGORIES )

print(CATEGORIES_MAP)

def get_latlon(wayid):
    df_sel = df[df["wayid"]==wayid]
    if len(df_sel) > 0:

        try:
            lat = float(df_sel["lat"])
            lon = float(df_sel["lon"])
        except:
            lat = float(df_sel.iloc[0]["lat"])
            lon = float(df_sel.iloc[0]["lon"])

    else:
        raise Exception("Way ID not found")

    return lat, lon

def get_wayid(filename):
    try:
        wayid = int(filename.split(".")[0])
    except:
        wayid = int(filename.split(".")[1])

    return wayid 

def generate_annotation(inputfolder, outfolder):
    '''
    crop the images 
    '''
    input_file = os.path.join(inputfolder, "via_region_data.json")
    output_file = os.path.join(outfolder, "annotation.json")

    coco = {"info": {"contributor": "stepenlee@umass", "about": "Building Dataset", 
            "date_created": "01/04/2018", "description": "Roof Orientation", 
            "url": "", "version": "1.0", "year": 2018},
            "categories":[],
            "images":[],
            "annotations":[]
            }

    
    # import the via_region_data.json
    with open(input_file, 'r') as fp:
        via_data = json.load(fp)


    categories = []; 
    images = [];
    annotations = [];
    annotation_id = 300000
    image_id = 100
    width = 512; height = 512; zoom = 20;

    for i, key in enumerate(via_data.keys()):
        filename = via_data[key]["filename"]
        
        wayid = get_wayid(filename)
        lat, lon = get_latlon(wayid)

        # create image obj
        image_id += 1
        image_obj ={"id": image_id, "file_name": filename, "width": width, "height": height, "lat": lat, "lon": lon, zoom:zoom}
        images.append(image_obj)

        # create annnotatoins
        for k, region in enumerate(via_data[key]["regions"].keys()):
            if (len(via_data[key]["regions"][region]["shape_attributes"]["all_points_x"]) <=2):
                continue

            annotation_id += 1
            # elimiate the segments or regions not in the bbox
            xys = zip(via_data[key]["regions"][region]["shape_attributes"]["all_points_x"],via_data[key]["regions"][region]["shape_attributes"]["all_points_y"])
            segmentation = []
            for x, y in xys:
                segmentation.append(x)
                segmentation.append(y)

            area = geom.Polygon(xys).area
            annotation_obj = {"id":annotation_id, "area":area, "image_id":image_id, "segmentation":[copy.deepcopy(segmentation)], "iscrowd":0}
            class_name = via_data[key]["regions"][region]["region_attributes"]["building"]
            
            # add annotations
            annotation_obj["category_id"] = CATEGORIES_MAP[class_name] ## categories_map[class_name]
            annotations.append(annotation_obj)
        
        coco["categories"] = CATEGORIES
        coco["annotations"] = annotations
        coco["images"] = images

    with open(output_file, 'w') as fp:
        json.dump(coco, fp)
        print("Saving.", output_file)

    
if __name__ == "__main__":
    
    inputfolder = "../data/deeproof-aug/"
    outfolder = "../data/deeproof-aug/"
    
    print("generating annotation for ", inputfolder)
    generate_annotation(inputfolder + "/test", outfolder + "/test")
