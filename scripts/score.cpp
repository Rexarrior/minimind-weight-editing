#include <llama.h>
#include <ggml-backend.h>
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using Clock=std::chrono::steady_clock;
struct Row { std::vector<int32_t> ids; int skip; };
int main(int argc,char **argv) {
    if(argc!=6) return 1;
    auto start=Clock::now();
    const int capacity=std::stoi(argv[4]), maxseq=std::stoi(argv[5]);
    ggml_backend_load_all();
    llama_backend_init();
    auto mp=llama_model_default_params(); mp.n_gpu_layers=99;
    auto *model=llama_model_load_from_file(argv[1],mp);
    if(!model) return 2;
    auto cp=llama_context_default_params();cp.n_ctx=capacity;cp.n_batch=capacity;cp.n_ubatch=capacity;cp.n_seq_max=maxseq;cp.kv_unified=true;cp.n_threads=8;cp.n_threads_batch=8;
    auto *ctx=llama_init_from_model(model,cp); if(!ctx) return 3;
    auto batch=llama_batch_init(capacity,0,1);
    const int vocab=llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> tmp(vocab);
    std::ifstream input(argv[2],std::ios::binary);std::ofstream output(argv[3]);
    if(!input||!output) return 4;
    output<<std::setprecision(12);
    long nrows=0,ntokens=0,last=0;
    Row pending;
    auto read=[&](Row& row){
        int32_t len,skip;
        if(!input.read(reinterpret_cast<char*>(&len),4)) {if(input.eof() && input.gcount()==0) return false; throw std::runtime_error("incomplete length");}
        if(!input.read(reinterpret_cast<char*>(&skip),4) || len<2 || skip<1 || skip>=len || len-1>capacity) throw std::runtime_error("unsupported row length "+std::to_string(len));
        row.ids.resize(len);row.skip=skip;
        if(!input.read(reinterpret_cast<char*>(row.ids.data()),len*4)) throw std::runtime_error("incomplete record");
        return true;
    };
    while(!pending.ids.empty() || read(pending)) {
        std::vector<Row> rows;int used=0;
        do {
            if(used+(int)pending.ids.size()-1>capacity) break;
            used+=pending.ids.size()-1;rows.emplace_back(std::move(pending));pending=Row{};
        } while((int)rows.size()<maxseq && read(pending));
        llama_memory_clear(llama_get_memory(ctx),false);
        batch.n_tokens=0;
        for(int s=0;s<(int)rows.size();++s) for(int i=0;i+1<(int)rows[s].ids.size();++i) {
            int j=batch.n_tokens++;batch.token[j]=rows[s].ids[i];batch.pos[j]=i;
            batch.n_seq_id[j]=1;batch.seq_id[j][0]=s;batch.logits[j]=(i+1>=rows[s].skip);
        }
        if(llama_decode(ctx,batch)!=0) throw std::runtime_error("decode failed");
        int j=0;
        for(const auto& row:rows) {
            double loss=0;int count=0;
            for(int i=1;i<(int)row.ids.size();++i,++j) {
                if(i<row.skip)continue;
                const float *p=llama_get_logits_ith(ctx,j);
                if(!p || row.ids[i]<0 || row.ids[i]>=vocab) throw std::runtime_error("bad logits/token");
                float top,sum;vDSP_maxv(p,1,&top,vocab);float neg=-top;
                vDSP_vsadd(p,1,&neg,tmp.data(),1,vocab);vvexpf(tmp.data(),tmp.data(),&vocab);
                vDSP_sve(tmp.data(),1,&sum,vocab);loss+=(double)top-p[row.ids[i]]+std::log((double)sum);++count;
            }
            if(!std::isfinite(loss)) throw std::runtime_error("nonfinite likelihood");
            output<<count<<" "<<loss<<"\n";++nrows;ntokens+=count;
        }
        if(ntokens-last>=100000) {output.flush();std::cerr<<"progress "<<nrows<<" "<<ntokens<<" "<<std::chrono::duration<double>(Clock::now()-start).count()<<"\n";last=ntokens;}
    }
    output.flush();llama_batch_free(batch);llama_free(ctx);llama_model_free(model);llama_backend_free();
    std::cerr<<"finished "<<nrows<<" "<<ntokens<<" "<<std::chrono::duration<double>(Clock::now()-start).count()<<"\n";
}
