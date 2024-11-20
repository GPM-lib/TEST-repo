
# Artifact for PPoPP'25 paper 
> *GLUMIN: Fast Connectivity Check Based on LUTs For Efficient Graph Pattern Mining.* 

## 1.1. Download datasets.

+ Download datasets.
Datasets are available [here](https://drive.google.com/drive/folders/1ZV5oeyJfCV922bwwKoWfe6SLwIoiaNOG?usp=sharing).  
+ `/dataset_table1` include all datasets shown in Table 1.
+ `/dataset_135` and `/dataset_table1` include all 135 datasets to test Figure 8.  


## 1.2. Compile implementation.
```
mkdir bin && make
```

# 2. Run initial test experiment with basic APIs.
## 2.1. Run baseline G<sup>2</sup>Miner.
>Usage: ./bin/pattern_gpu_GM <graph_path> <pattern_id>  
>Support Patterns in Figure 7: P1-P3, P6-P22

```
./bin/pattern_gpu_GM ./datasets/cit-Patents/graph 1
./bin/pattern_gpu_GM ./datasets/youtube/graph 13
./bin/pattern_gpu_GM ./datasets/soc-pokec/graph 17
```

`Clique patterns (P4,P5,P23,P24)`, Correspond to 4,5,6,7-Clique`
>Usage: ./bin/clique_GM <graph_path> <clique_size>  

```
./bin/clique_GM ./datasets/livej/graph 5
```

## 2.2. Run G<sup>2</sup>Miner + LUT.
>Usage: ./bin/pattern_gpu_GM_LUT <graph_path> <pattern_id>  
>Support Patterns in Figure 7: P1-P3, P6-P22

```
./bin/pattern_gpu_GM_LUT ./datasets/cit-Patents/graph 1
./bin/pattern_gpu_GM_LUT ./datasets/youtube/graph 13
./bin/pattern_gpu_GM_LUT ./datasets/soc-pokec/graph 17
```

`Clique patterns (P4,P5,P23,P24)`, Correspond to 4,5,6,7-Clique`
>Usage: ./bin/clique_GM_LUT <graph_path> <clique_size>  

```
./bin/clique_GM_LUT ./datasets/livej/graph 5
```

## 2.3. Run GraphFold & GraphFold + LUT.
>Usage: ./bin/pattern_gpu_GF_LUT <graph_path> <pattern_id>  
>Support Patterns in Figure 7: P1, P10, P13.  
>GraphFold + LUT: <pattern_id> = 1, 10, 13.  
>GraphFold: <pattern_id> = 2, 11, 14

```
./bin/pattern_gpu_GF_LUT ./datasets/cit-Patents/graph 1
```

`Clique patterns (P5)`, Correspond to 5-Clique`
>Usage: ./bin/clique_GM_LUT <graph_path> <switch_id>   
><switch_id> = 1 for GraphFold, 2 for GraphFold + LUT

```
./bin/clique_GF_LUT ./datasets/livej/graph 2
```

## 2.4. Run AutoMine & AntoMine + LUT.
>Usage: ./bin/automine_LUT <graph_path> <pattern_id>  
>Support Patterns in Figure 7: P1, P7, P10, P13, P15, P20  
>AutoMine + LUT: <pattern_id> = 1, 7, 10, 13, 15, 20.  
>AutoMine: <pattern_id> = 2, 8, 11, 14, 16, 21

```
./bin/automine_LUT ./datasets/cit-Patents/graph 1
```

## 2.5. Codegen

Codegen source: `codegen/*`

## Generated pattern
Codegen pattern yml example: `codegen/codegen/patterns`

<!-- # 3. Run GLumin in docker.
## 3.1 Launch the GLumin docker
```
cd docker 
./build.sh
```

## 3.2 Launch the GLumin docker and recompile, 
+ The compiled exectuable will be located under `GPM-artifact/`.
```
cd docker 
./launch.sh
cd GPM-artifact && mkdir bin && make
``` -->
