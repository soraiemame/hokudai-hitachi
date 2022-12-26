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
    int T_weather,N_weather;
    vector<vector<ld>> weather_graph;
    vector<int> c_weather;
    ld P_m,R_m,alpha;
    vector<vector<ld>> weather_forecast;

    int today_weather;
    
    class Worker{
        public:
            int N_type,L,pos,pos2,dist;
            set<int> type;
    };

    vector<Worker> worker;

    class Job{
        public:
            int id,type,N,v,f;
            ld P,d_weather;
            vector<pair<int,ll>> reward; 
            vector<ll> score;
            vector<int> depend;
            vector<ld> weather_task;
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
            cin >> job[i].id >> job[i].type >> job[i].N >> job[i].v >> job[i].P >> job[i].d_weather >> job[i].f;
            job[i].id--;
            job[i].type--;
            job[i].v--;
            if (job[i].f==1) selected_jobs.push_back(i);
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
        cin >> T_weather >> N_weather;
        weather_graph.resize(N_weather);
        for (int i=0;i<N_weather;i++){
            weather_graph[i].resize(N_weather);
            for (int j=0;j<N_weather;j++){
                cin >> weather_graph[i][j];
            }
        }
        c_weather.resize(N_weather);
        for (int i=0;i<N_weather;i++) cin >> c_weather[i];
        for (int i=0;i<N_job;i++){
            for (int j=0;j<N_weather;j++) job[i].weather_task.push_back(pow((ld)1-job[i].d_weather,c_weather[j]));
        }
        weather_forecast.resize(T_max/T_weather);
        cin >> P_m >> R_m >> alpha;
        for (int i=0;i<T_max/T_weather;i++){
            int t_i;
            cin >> t_i;
            weather_forecast[i].resize(N_weather);
            for (int j=0;j<N_weather;j++){
                cin >> weather_forecast[i][j];
            }
        }
    }

    void interact(int now){
        cin >> today_weather;
        today_weather--;
        int n_job;
        cin >> n_job;
        for (int i=0;i<n_job;i++){
            int f;
            cin >> f;
            f--;
            cin >> job[f].N;
        }
        for (int i=0;i<N_worker;i++){
            int id;
            cin >> id;
            id--;
            cin >> worker[id].pos >> worker[id].pos2 >> worker[id].dist;
            worker[id].pos--;
            worker[id].pos2--;
        }
        if (now%T_weather==0){
            for (int i=now/T_weather;i<T_max/T_weather;i++){
                int t_i;
                cin >> t_i;
                for (int j=0;j<N_weather;j++) cin >> weather_forecast[i][j];
            }
        }
    }
} IOServer;

struct solver{
    //determine worker's job
    vector<int> worker_job;
    void init(){
        worker_job.resize(IOServer.N_worker,-1);
        for (int i=0;i<IOServer.selected_jobs.size();i++){
            for (int j:IOServer.job[IOServer.selected_jobs[i]].depend){
                if (IOServer.job[j].f==0) {
                    IOServer.job[j].f=1;
                    IOServer.selected_jobs.push_back(j);
                }
            }
        }
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
        set<int> sj;
        for(auto j:IOServer.selected_jobs)sj.emplace(j);
        cout << sj.size() << " ";
        for (int i:sj) cout << i+1 << " ";
        cout << endl;
    }

    //determine action
    void solve(){
        for (int i=0;i<IOServer.T_max;i++){
            IOServer.interact(i);
            if (i==0){
                cout << IOServer.N_worker << endl;
                for (int j=0;j<IOServer.N_worker;j++) cout << j+1 << " ";
                cout << endl;
                for (int j=0;j<IOServer.N_worker;j++){
                    for (int k=0;k<IOServer.T_max;k++) cout << worker_job[j]+1 << " ";
                    cout << endl;
                }
            } else {
                cout << 0 << endl;
                cout << endl;
            }
            for (int j=0;j<IOServer.N_worker;j++){
                if (IOServer.worker[j].pos==IOServer.job[worker_job[j]].v && IOServer.worker[j].dist==0){
                    if (IOServer.job[worker_job[j]].N>0 && 
                    IOServer.job[worker_job[j]].score[i+1]-IOServer.job[worker_job[j]].score[i] &&
                    IOServer.worker[j].L*IOServer.job[worker_job[j]].weather_task[IOServer.today_weather]>0){
                        cout << "execute " << worker_job[j]+1 << " " << min(IOServer.job[worker_job[j]].N,(int)(IOServer.worker[j].L*IOServer.job[worker_job[j]].weather_task[IOServer.today_weather])) << endl;
                    } else {
                        cout << "stay" << endl;
                    }
                } else {
                    cout << "move " << IOServer.job[worker_job[j]].v+1 << endl;
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