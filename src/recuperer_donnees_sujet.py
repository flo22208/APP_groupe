def retrieve_subject_data(num_sujet):
    '''
    Fonction permettant de récupérer les données associées à un sujet
    
    Args: int : le numéro du sujet
    
    Return: - string : le chemin vers la vidéo
            - string : le chemin vers le fichier gaze
            - string : le chemin vers les paramètres de la caméra
    '''
    sujet_ids = ["sujet1_f-42e0d11a", "sujet2_f-835bf855", 
                 "sujet3_m-84ce1158", "sujet4_m-fee537df", 
                 "sujet5_m-671cf44e", "sujet6_m-0b355b51"]
    video_names = ["e0b2c246_0.0-138.011", "b7bd6c34_0.0-271.583",
                   "422f10f2_0.0-247.734", "2fb8301a_0.0-71.632",
                   "585d8df7_0.0-229.268", "429d311a_0.0-267.743"]
    
    sujet_id = sujet_ids[num_sujet-1]
    video_name = video_names[num_sujet-1]
    
    video_file = f"AcquisitionsEyeTracker/{sujet_id}/{video_name}.mp4"
    gaze_file = f"AcquisitionsEyeTracker/{sujet_id}/gaze.csv"
    camera_parameters_file =  f"AcquisitionsEyeTracker/{sujet_id}/scene_camera.json"
    
    return video_file, gaze_file, camera_parameters_file

a,b,c = retrieve_subject_data(1)

print(a)
print(b)
print(c)