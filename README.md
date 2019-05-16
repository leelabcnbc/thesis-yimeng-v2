# thesis-yimeng-v2
good parts of thesis-yimeng-v1, better refactoring.

`$ROOT` refers to repo root.

## set up toolchain (only once)

On May 16, 2019 (Eastern Time), I ran this on my MacBook Pro (steps 1-3), and
`yimengzh2080ti.cnbc.cmu.edu` (steps 3-6).

1. under `$ROOT/toolchain/standard`, run
    ~~~
    docker build . -t leelabcnbc/yimeng-thesis-v2:standard
    ~~~
2. then run `docker save --output yimeng-thesis-v2.tar leelabcnbc/yimeng-thesis-v2:standard` to save the image to a tar file.
   last time I ran it, MD5 was `20640abe170a90a7d323165ed8de0c5b`
3. then upload that file to somewhere with singularity
   (tested on 2.6.1, which is compatible with 2.5.0 on CNBC psych-o cluster)
4. run `sudo docker load -i yimeng-thesis-v2.tar` to get it.
5. check <https://github.com/sylabs/singularity/issues/1537#issuecomment-388642244>
   and <https://github.com/sylabs/singularity/issues/1537#issuecomment-402823767>
    ~~~
    # Start a docker registry
    docker run -d -p 5000:5000 --restart=always --name registry registry:2
    # Push local docker container to it
    docker tag leelabcnbc/yimeng-thesis-v2:standard localhost:5000/yimeng-thesis-v2:standard
    docker push localhost:5000/yimeng-thesis-v2:standard
    # build to get `yimeng-thesis-v2.simg`
    sudo SINGULARITY_NOHTTPS=1 /opt/singularity-2.6.1/bin/singularity build yimeng-thesis-v2.simg docker://localhost:5000/yimeng-thesis-v2:standard
    ~~~
   last time I ran this, I get md5 of `34da8d5eac5297d9fafa5e8be3c635f0`.
6. rename the file according to the MD5. For me, it's
   `yimeng-thesis-v2_34da8d5eac5297d9fafa5e8be3c635f0.simg`.

## run toolchain

### `yimengzh2080ti.cnbc.cmu.edu`

```
/opt/singularity-2.6.1/bin/singularity shell --nv -B /data2:/my_data -B /data1:/my_data_2 -B /run:/run ~/toolchain/yimeng-thesis-v2_34da8d5eac5297d9fafa5e8be3c635f0.simg
```

### CNBC cluster

TBD
