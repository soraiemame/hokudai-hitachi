#include <vector>
#include <utility>
#include <set>
#include <queue>
#include <functional>
#include <iostream>
#include <cmath>
using namespace std;
#define ll long long
#define ld long double



//Assign one job to each worker and execute the job

struct IOServer {
    vector<int> selected_jobs;
    int T_max;
    int V,E;
    vector<vector<pair<int,int>>> edge;
    vector<vector<int>> dist;
    int N_worker,N_job;
    
    class Worker{
        public:
            int N_type,L,pos,pos2,dist;
            set<int> type;
    };

    vector<Worker> worker;

    class Job{
        public:
            int id,type,N,v;
            vector<pair<int,ll>> reward; 
            vector<ll> score;
            vector<int> depend;
    };

    vector<Job> job;


    //caliculate distance
    void dij(){
        dist.resize(V);
        for (int i=0;i<V;i++){
            dist[i].resize(V);
            for (int j=0;j<V;j++) dist[i][j]=1e9;
            dist[i][i]=0;
            priority_queue<pair<int,int>, std::vector<pair<int,int>>, std::greater<pair<int,int>>> q;
            q.push({0,i});
            while(!q.empty()){
                pair<int,int> t=q.top();
                q.pop();
                if (dist[i][t.second]!=t.first) continue;
                for (auto e:edge[t.second]){
                    if (dist[i][e.first]>dist[i][t.second]+e.second){
                        dist[i][e.first]=dist[i][t.second]+e.second;
                        q.push({dist[i][e.first],e.first});
                    }
                }
            }
        }
    }

	void input(){
        cin >> T_max;
        cin >> V >> E;
        edge.resize(V);
        for (int i=0;i<E;i++){
            int f,g,h;
            cin >> f >> g >> h;
            f--;
            g--;
            edge[f].push_back({g,h});
            edge[g].push_back({f,h});
        }
        dij();
        cin >> N_worker;
        worker.resize(N_worker);
        for (int i=0;i<N_worker;i++){
            cin >> worker[i].pos >> worker[i].L >> worker[i].N_type;
            for (int j=0;j<worker[i].N_type;j++){
                int f;
                cin >> f;
                f--;
                worker[i].type.insert(f);
            }
            worker[i].pos--;
            worker[i].pos2=worker[i].pos;
            worker[i].dist=0;
        }
        cin >> N_job;
        job.resize(N_job);
        for (int i=0;i<N_job;i++){
            cin >> job[i].id >> job[i].type >> job[i].N >> job[i].v;
            job[i].id--;
            job[i].type--;
            job[i].v--;
            int N_reward;
            cin >> N_reward;
            for (int j=0;j<N_reward;j++){
                int f,g;
                cin >> f >> g;
                f--;
                job[i].reward.push_back({f,g});
            }
            job[i].reward.push_back({(int)T_max+2,job[i].reward[N_reward-1].second});
            job[i].score.resize(T_max+2,0);
            job[i].score[0]=0;
            int it=0;
            for (int j=0;j<T_max+1;j++){
                if (it==0) {
                    job[i].score[j+1]=0;
                    if (j>=job[i].reward[it].first){
                        it++;
                    }
                    continue;
                }
                ll now_score=job[i].reward[it-1].second+(j-job[i].reward[it-1].first)*(job[i].reward[it].second-job[i].reward[it-1].second)/(job[i].reward[it].first-job[i].reward[it-1].first);
                job[i].score[j+1]=job[i].score[j]+now_score;
                if (j>=job[i].reward[it].first){
                    it++;
                }
            }
            int N_depend;
            cin >> N_depend;
            for (int j=0;j<N_depend;j++){
                int f;
                cin >> f;
                f--;
                job[i].depend.push_back(f);
            }
        }
    }
} IOServer;

struct solver{
    //determine worker's job
    vector<int> worker_job;
    void init(){
        worker_job.resize(IOServer.N_worker,-1);
        vector<int> is_selected(IOServer.N_job,0);
        for (int i=0;i<IOServer.N_worker;i++){
            for (int j=0;j<IOServer.N_job;j++){
                if (is_selected[j]) continue;
                if (IOServer.worker[i].type.find(IOServer.job[j].type)==IOServer.worker[i].type.end()) continue;
                if (IOServer.job[j].depend.size()>0) continue;
                is_selected[j]=1;
                worker_job[i]=j;
                IOServer.selected_jobs.push_back(j);
                break;
            }
        }
    }

    //determine action
    void solve(){
        for (int i=0;i<IOServer.T_max;i++){
            for (int j=0;j<IOServer.N_worker;j++){
                if (IOServer.worker[j].pos==IOServer.job[worker_job[j]].v && IOServer.worker[j].dist==0){
                    if (IOServer.job[worker_job[j]].N>0 && 
                    IOServer.job[worker_job[j]].score[i+1]-IOServer.job[worker_job[j]].score[i]){
                        cout << "execute " << worker_job[j]+1 << " " << min(IOServer.job[worker_job[j]].N,(int)IOServer.worker[j].L) << endl;
                        IOServer.job[worker_job[j]].N-=min(IOServer.job[worker_job[j]].N,(int)IOServer.worker[j].L);
                    } else {
                        cout << "stay" << endl;
                    }
                } else {
                    if (IOServer.worker[j].pos!=IOServer.job[worker_job[j]].v) {
                        IOServer.worker[j].dist=IOServer.dist[IOServer.worker[j].pos][IOServer.job[worker_job[j]].v];
                        IOServer.worker[j].pos=IOServer.job[worker_job[j]].v;
                    }
                    cout << "move " << IOServer.job[worker_job[j]].v+1 << endl;
                    IOServer.worker[j].dist--;
                }
            }
        }
        long long score;
        cin>>score; // read score to avoid TLE
    }
} solver;

int main(int argc, char *argv[] ){
    IOServer.input();
    solver.init();
    solver.solve();
}