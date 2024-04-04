from Metric import eval_main, postprocess_MSD_main

if __name__ == '__main__':
    infer_bathpath = '.../fold0/missing_modality'
    label_path = '.../labels'
    Post = True

    infer_bathpath += '/missing_'
    for i in range(14, -1, -1):
        try:
            infer_path = infer_bathpath+str(i)
            post_path = infer_path + '_post' if Post else infer_path
            if Post:
                postprocess_MSD_main(infer_path, post_path, 500, ET=4)
            eval_main(infer_dirpath=post_path, 
                      label_dirpath=label_path,
                      region_list=[
                          [1,2,3,4],
                          [1,3,4],
                          [4]
                      ])
        except Exception as e:
            print(e)