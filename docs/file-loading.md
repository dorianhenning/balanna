# File Loading speeds with different File Formats

### Json load from file
```
Timings
ID                                        #Measurements     Mean          Max           Cumulative    Percentage
load_scene_from_json                      401               0.02279787s   0.13157415s   9.14194417s   52.42%
cylinder                                  11629             0.00039315s   0.11832547s   4.57192278s   26.21%
point_cloud                               12431             0.00016401s   0.04467106s   2.03884315s   11.69%
frame                                     802               0.00179561s   0.00223112s   1.44007850s   8.26%
trajectory                                802               0.00030764s   0.05406260s   0.24672675s   1.41%
sphere                                    1                 0.00174856s   0.00174856s   0.00174856s   0.01%
```

### Gzip load from file
```
Timings
ID                                        #Measurements     Mean          Max           Cumulative    Percentage
load_scene_from_json                      401               0.02111126s   0.13129783s   8.46561694s   50.26%
cylinder                                  11629             0.00039469s   0.11939788s   4.58983135s   27.25%
point_cloud                               12431             0.00016874s   0.04472232s   2.09762669s   12.45%
frame                                     802               0.00179289s   0.00230002s   1.43789721s   8.54%
trajectory                                802               0.00031416s   0.05542707s   0.25195432s   1.50%
sphere                                    1                 0.00166178s   0.00166178s   0.00166178s   0.01%
```

### CBOR load from file
```
Timings
ID                                        #Measurements     Mean          Max           Cumulative    Percentage
load_scene_from_json                      401               0.01331337s   0.11966467s   5.33866215s   39.15%
cylinder                                  11629             0.00039362s   0.10413170s   4.57745790s   33.57%
point_cloud                               12431             0.00015816s   0.00362539s   1.96604085s   14.42%
frame                                     802               0.00178729s   0.00216126s   1.43340802s   10.51%
trajectory                                802               0.00039384s   0.08667588s   0.31585693s   2.32%
sphere                                    2                 0.00178540s   0.00187755s   0.00357080s   0.03%
```

### Large gzip file
```
Timings
ID                                        #Measurements     Mean          Max           Cumulative    Percentage
load_scene_from_gzip                      1                 6.55068946s   6.55068946s   6.55068946s   44.84%
cylinder                                  12238             0.00036284s   0.00076127s   4.44047976s   30.40%
point_cloud                               13082             0.00015045s   0.00285149s   1.96821165s   13.47%
frame                                     844               0.00178372s   0.00209355s   1.50545597s   10.31%
trajectory                                844               0.00016577s   0.00044084s   0.13990903s   0.96%
sphere                                    2                 0.00175321s   0.00179911s   0.00350642s   0.02%
```