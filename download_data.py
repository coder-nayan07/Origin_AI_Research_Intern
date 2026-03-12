from roboflow import Roboflow


rf = Roboflow(api_key="tm3sZ8ebG8ZjGyKa8nZV")
project = rf.workspace("creator-mnadn").project("cracks-3ii36-ghpzn")
version = project.version(1)
dataset = version.download("coco")


rf = Roboflow(api_key="tm3sZ8ebG8ZjGyKa8nZV")
project = rf.workspace("objectdetect-pu6rn").project("drywall-join-detect")
version = project.version(2)
dataset = version.download("coco")
                
                
