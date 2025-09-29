from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility


api = HubApi()
api.login()


owner_name = 'nev8rz'   
model_name = 'NekoChat'   
model_id = f"{owner_name}/{model_name}"


# api.create_model(
#      model_id,
#      visibility=ModelVisibility.PUBLIC,
#      license=Licenses.APACHE_V2,
#      chinese_name=f"NekoChat"
#     )

# 上传模型
api.upload_folder(
    repo_id=f"{owner_name}/{model_name}",
    folder_path='./models/NekoChat',    
    commit_message='NekoChat',   
)