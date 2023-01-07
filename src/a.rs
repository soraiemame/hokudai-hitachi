use proconio::{input, marker::Usize1, source::line::LineSource};
use std::{io::{BufReader, Stdin}};
use rand::prelude::*;

fn main() {
    get_time();
    let mut stdin =
        proconio::source::line::LineSource::new(std::io::BufReader::new(std::io::stdin()));
    let input = Input::from_stdin(&mut stdin);
    eprintln!("{:?}",get_time());
    let solver = Solver::new();
    // let output = solver.solve(input);
    let output = solver.solve2(input);
    eprintln!("{:?}",get_time());
    print!("{}",output.to_string());
    input! {
        from &mut stdin,
        score: u64
    }
    println!("{}", score);
    eprintln!("{:?}",get_time());
}

#[derive(Debug)]
struct Input {
    t_max: usize,
    v: usize,
    e: usize,
    graph: Vec<Vec<(usize, u32)>>,
    n_worker: usize,
    workers: Vec<Worker>,
    n_job: usize,
    jobs: Vec<Job>,
}

impl Input {
    fn from_stdin(mut stdin: &mut LineSource<BufReader<Stdin>>) -> Self {
        input! {
            from &mut stdin,
            t_max: usize,
            v: usize,e: usize,
            edges: [(Usize1,Usize1,u32);e],
        }
        let mut graph = vec![vec![]; v];
        for (a, b, c) in edges {
            graph[a].push((b, c));
            graph[b].push((a, c));
        }
        input! {
            from &mut stdin,
            n_worker: usize
        }
        let mut workers = vec![];
        workers.reserve(n_worker);
        for _ in 0..n_worker {
            input! {
                from &mut stdin,
                v: Usize1,
                l_max: u32,
                types: [Usize1]
            }
            workers.push(Worker::new(v, l_max, types));
        }
        input! {
            from &mut stdin,
            n_job: usize,
        }
        let mut jobs = vec![];
        jobs.reserve(n_job);
        for _ in 0..n_job {
            input! {
                from &mut stdin,
                id: Usize1,
                job_type: Usize1,
                n_task: u32,
                v: Usize1,
                reward: [(u32,u64)],
                deps: [Usize1],
            }
            jobs.push(Job::new(id, job_type, n_task, v, reward, deps));
        }
        Self {
            t_max,
            v,
            e,
            graph,
            n_worker,
            workers,
            n_job,
            jobs,
        }
    }
}

#[derive(Debug, Clone)]
struct Worker {
    pos: usize,
    // 現時点で目指している頂点 or !0
    pos2: usize,
    dist: u32,
    l_max: u32,
    types: u8,
}
impl Worker {
    fn new(v: usize, l_max: u32, pre_types: Vec<usize>) -> Self {
        let mut types = 0;
        for t in pre_types {
            types |= 1 << t;
        }
        Self {
            pos: v,
            pos2: !0,
            dist: 0,
            l_max,
            types,
        }
    }
    fn can_do(&self, job: &Job) -> bool {
        (self.types >> job.job_type & 1) != 0
    }
    fn move_to(&mut self, t: usize, dist_pp: &Vec<Vec<u32>>) {
        if self.pos2 == !0 {
            self.pos2 = t;
        }
        // assert_ne!(self.pos,!0);
        // assert_ne!(self.pos2,!0);
        // assert_ne!(t,!0);
        if dist_pp[self.pos][t] < dist_pp[self.pos2][t] {
            self.dist -= 1;
        } else {
            self.dist += 1;
        }
        if self.dist == dist_pp[self.pos][self.pos2] {
            self.pos = self.pos2;
            self.pos2 = !0;
            self.dist = 0;
        } else if self.dist == 0 {
            self.pos2 = !0;
        }
    }
}

#[derive(Debug, Clone)]
struct Job {
    id: usize,
    job_type: usize,
    n_task: u32,
    v: usize,
    // (start,end)
    start: usize,
    end: usize,
    // reward: Vec<(u32, u64)>,
    reward: Vec<u64>,
    deps: Vec<usize>,
}
impl Job {
    fn new(
        id: usize,
        job_type: usize,
        n_task: u32,
        v: usize,
        raw_reward: Vec<(u32, u64)>,
        deps: Vec<usize>,
    ) -> Self {
        let start = raw_reward[0].0 as usize;
        let end = raw_reward[raw_reward.len() - 1].0 as usize;
        let mut reward = vec![];
        for i in 0..raw_reward.len() - 1 {
            for t in raw_reward[i].0..raw_reward[i + 1].0 {
                if t == 0 {
                    continue;
                }
                let y_prev = raw_reward[i].1;
                let y_next = raw_reward[i + 1].1;
                let t_prev = raw_reward[i].0;
                let t_next = raw_reward[i + 1].0;
                reward.push(
                    (y_next - y_prev) * (t - t_prev) as u64 / (t_next - t_prev) as u64 + y_prev
                );
            }
        }
        Self {
            id,
            job_type,
            n_task,
            v,
            start,
            end,
            reward,
            deps,
        }
    }
    fn get_reward(&self,t: usize) -> u64 {
        if t <= self.start || self.end <= t {0} else {self.reward[t - self.start - 1]}
    }
    fn can_finish(&self, start_turn: usize, l_max: u32) -> bool {
        let end_turn = start_turn + ((self.n_task + l_max - 1) / l_max as u32) as usize;
        end_turn < self.end
    }
    fn can_do(&self, turn: usize) -> bool {
        self.start < turn && turn < self.end
    }
}

enum Action {
    Stay,
    Move(usize),
    Execute(usize, u32),
}

// inputで状態を表すもののみ保持
#[derive(Debug, Clone)]
struct State {
    turn: usize,
    worker_pos: Vec<(usize,usize,u32)>,
    job_remain: Vec<u32>,
    reward_got: Vec<u64>,
}

impl State {
    fn new(
        turn: usize,
        worker_pos: Vec<(usize,usize,u32)>,
        job_remain: Vec<u32>,
        reward_got: Vec<u64>,
    ) -> Self {
        Self {
            turn,
            worker_pos,
            job_remain,
            reward_got,
        }
    }
    fn from_input(input: &Input) -> Self {
        let mut worker_pos = vec![];
        for i in 0..input.n_worker {
            worker_pos.push((input.workers[i].pos,input.workers[i].pos2,input.workers[i].dist));
        }
        let mut job_remain = vec![];
        for i in 0..input.n_job {
            job_remain.push(input.jobs[i].n_task);
        }
        let reward_got = vec![0;input.n_job];
        Self {
            turn: 0,
            worker_pos,
            job_remain,
            reward_got
        }
    }
    // 現時点で獲得した報酬の量、各ワーカーと各ジョブの位置関係を得点にする
    fn score(&self, input: &Input) -> u64 {
        let mut res = 0;
        // cur score
        for i in 0..input.n_job {
            if self.job_remain[i] == 0 {
                res += self.reward_got[i];
            }
        }
        res
    }
    // fn next_states(&self, input: &Input) -> Vec<(Vec<Action>, State)> {
    //     vec![]
    // }
    fn apply_action(&mut self,input: &Input,dist_pp: &Vec<Vec<u32>>,idx: usize,action: Action) {
        match action {
            Action::Stay => {},
            Action::Move(p) => {
                if self.worker_pos[idx].1 == !0 {
                    self.worker_pos[idx].1 = p;
                }
                // assert_ne!(self.pos,!0);
                // assert_ne!(self.pos2,!0);
                // assert_ne!(t,!0);
                if dist_pp[self.worker_pos[idx].0][p] < dist_pp[self.worker_pos[idx].1][p] {
                    self.worker_pos[idx].2 -= 1;
                } else {
                    self.worker_pos[idx].2 += 1;
                }
                if self.worker_pos[idx].2 == dist_pp[self.worker_pos[idx].0][self.worker_pos[idx].1] {
                    self.worker_pos[idx].0 = self.worker_pos[idx].1;
                    self.worker_pos[idx].1 = !0;
                    self.worker_pos[idx].2 = 0;
                } else if self.worker_pos[idx].2 == 0 {
                    self.worker_pos[idx].1 = !0;
                }
            },
            Action::Execute(i, a) => {
                self.job_remain[i] -= a;
                self.reward_got[i] += input.jobs[i].get_reward(self.turn) * a as u64;
            }
        };
    }
    fn tick(&mut self) {self.turn += 1;}
}

struct Solver;

impl Solver {
    fn new() -> Self {
        Self {}
    }
    fn dijkstra(&self, input: &Input) -> (Vec<Vec<u32>>, Vec<Vec<usize>>) {
        use std::{cmp::Reverse, collections::BinaryHeap};
        let mut res = vec![];
        res.reserve(input.v);
        let mut par = vec![vec![!0usize; input.v]; input.v];
        for i in 0..input.v {
            let mut dist = vec![!0u32; input.v];
            dist[i] = 0;
            let mut que = BinaryHeap::new();
            que.push((Reverse(0), i));
            while let Some((Reverse(d), p)) = que.pop() {
                if dist[p] < d {
                    continue;
                }
                for &(a, c) in &input.graph[p] {
                    if dist[a] > dist[p] + c {
                        dist[a] = dist[p] + c;
                        que.push((Reverse(dist[a]), a));
                        par[i][a] = p;
                    }
                }
            }
            res.push(dist);
        }
        (res, par)
    }
    fn solve(self, mut input: Input) -> Output {
        let (dist_pp, par_pp) = self.dijkstra(&input);
        eprintln!("{:?}",get_time());
        let dist = |w: &Worker, t: usize| -> u32 {
            if w.pos2 == !0 {
                dist_pp[w.pos][t]
            } else {
                if dist_pp[w.pos][t] < dist_pp[w.pos2][t] {
                    dist_pp[w.pos][t] + w.dist
                } else {
                    dist_pp[w.pos][t] - (dist_pp[w.pos][w.pos2] - w.dist)
                }
            }
        };
        let mut res = vec![];
        res.reserve(input.t_max);

        let mut task_done = vec![false; input.n_job];
        for turn in 0..input.t_max {
            let mut turn_action = vec![];
            turn_action.reserve(input.t_max);
            let mut add_done = vec![];
            for i in 0..input.n_worker {
                let cur_worker = &input.workers[i];
                let jobs_cando = input.jobs.iter().filter(|j| {
                    let d = dist(&cur_worker, j.v);
                    !task_done[j.id]
                        && j.n_task != 0
                        && cur_worker.can_do(j)
                        && j.deps.iter().all(|&j2| task_done[input.jobs[j2].id])
                        && j.can_finish(turn + d as usize, cur_worker.l_max)
                });
                // let closest = jobs_cando.map(|j| (dist(&cur_worker, j.v), j.id)).min();
                // let closest = jobs_cando.map(|j| (j.reward[0].0, j.id)).min();
                let closest = jobs_cando
                    .map(|j| {
                        let d = dist(&cur_worker, j.v);
                        let arrive = turn + d as usize;
                        let wait = j.start.saturating_sub(arrive);
                        (wait + d as usize, j.id)
                    })
                    .min();
                if closest.is_none() {
                    turn_action.push(Action::Stay);
                    continue;
                }
                let closest = closest.unwrap().1;
                let closest_job = &input.jobs[closest];
                // do_work
                // if dist_pp[closest_job.v][cur_worker.pos] == 0 {
                if cur_worker.pos == closest_job.v && cur_worker.pos2 == !0 {
                    if !closest_job.can_do(turn) {
                        turn_action.push(Action::Stay);
                        continue;
                    }
                    let task_do = input.jobs[closest].n_task.min(cur_worker.l_max);
                    turn_action.push(Action::Execute(closest_job.id, task_do));

                    input.jobs[closest].n_task -= task_do;
                    if input.jobs[closest].n_task == 0 {
                        add_done.push(input.jobs[closest].id);
                    }
                }
                // move_to_duty
                else {
                    let p = cur_worker.pos;
                    if p == closest_job.v {
                        turn_action.push(Action::Move(p));
                        input.workers[i].move_to(p, &dist_pp);
                    } else {
                        turn_action.push(Action::Move(par_pp[closest_job.v][p]));
                        input.workers[i].move_to(par_pp[closest_job.v][p], &dist_pp);
                    }
                }
            }
            res.push(turn_action);
            for add in add_done {
                task_done[add] = true;
            }
        }
        Output::new(res)
    }
    // 各ワーカーごとにする仕事を割り当てて愚直にやった場合の報酬を焼きなます
    // ワーカーにジョブを追加する、ジョブを削除する、ジョブを消して新しいジョブを追加するが近傍
    fn solve2(self, mut input: Input) -> Output {
        let (dist_pp, par_pp) = self.dijkstra(&input);
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(432908);
        let mut res = vec![];
        let mut mx_score = 0;
        let mut selected = vec![vec![];input.n_worker];
        let mut used = vec![false;input.n_job];
        let mut uf = acl::Dsu::new(input.n_job);
        for i in 0..input.n_job {
            for &x in &input.jobs[i].deps {
                uf.merge(i, x);
            }
        }
        let groups = uf.groups();
        let mut ids = vec![0;input.n_job];
        for (i,g) in groups.iter().enumerate() {
            for k in g {
                ids[*k] = i;
            }
        }
        loop {
            let cur = get_time();
            if cur > 4.95 {
                break;
            }
            let t = rng.gen::<i32>() % 3;
            // delete add
            if t == 0 {
                continue;
            }
            // add
            else if t == 1 {
                let mut j_idx = rng.gen_range(0, input.n_job);
                let mut cnt = 0;
                while (used[j_idx] || input.jobs[j_idx].deps.len() != 0) && cnt < 10 {
                    j_idx = rng.gen_range(0, input.n_job);
                    cnt += 1;
                }
                if cnt == 10 {
                    continue;
                }
                let w_idx = rng.gen_range(0, input.n_worker);
                if !input.workers[w_idx].can_do(&input.jobs[j_idx]) {
                    continue;
                }
                selected[w_idx].push(j_idx);
                used[j_idx] = true;
                let (acts,score) = self.run(&input, &dist_pp,&par_pp, &selected);
                if chmax!(mx_score,score) {
                    res = acts;
                }
            }
            // delete
            else {
                continue;
                let w_idx = rng.gen_range(0, input.n_worker);
                if selected[w_idx].len() == 0 {
                    continue;
                }
                let j_idx = rng.gen_range(0, selected[w_idx].len());
                selected[w_idx].swap_remove(j_idx);
            }
        }
        Output::new(res)
    }
    fn run(&self,input: &Input,dist_pp: &Vec<Vec<u32>>,par_pp: &Vec<Vec<usize>>,jobs: &Vec<Vec<usize>>) -> (Vec<Vec<Action>>,u64) {
        let mut res = vec![];
        let mut score = 0;
        let mut cs = State::from_input(input);
        for t in 0..input.t_max {
            let mut turn_action = vec![];
            for i in 0..input.n_worker {
                let mn = jobs[i].iter().min_by_key(|&x| if t > input.jobs[*x].end {1 << 30}else{input.jobs[*x].start});
                if mn.is_none() {
                    turn_action.push(Action::Stay);
                    continue;
                }
                let mn = *mn.unwrap();
                if input.jobs[mn].v == cs.worker_pos[i].0 && cs.worker_pos[i].2 == 0 {
                    if input.jobs[mn].get_reward(t) == 0 || cs.job_remain[mn] == 0 {
                        turn_action.push(Action::Stay);
                        continue;
                    }
                    let a = input.workers[i].l_max.min(cs.job_remain[mn]);
                    turn_action.push(Action::Execute(mn, a));
                    cs.apply_action(input, dist_pp, i, Action::Execute(mn, a));
                    score += a as u64 * input.jobs[mn].get_reward(t);
                }
                else {
                    let p = cs.worker_pos[i].0;
                    if p == input.jobs[mn].v {
                        // eprintln!("up:{}",p);
                        turn_action.push(Action::Move(p));
                        cs.apply_action(input, dist_pp, i, Action::Move(p));
                    } else {
                        // eprintln!("down:{}",par_pp[input.jobs[mn].v][p]);
                        turn_action.push(Action::Move(par_pp[input.jobs[mn].v][p]));
                        cs.apply_action(input, dist_pp, i, Action::Move(par_pp[input.jobs[mn].v][p]));
                    }
                }
            }
            cs.tick();
            res.push(turn_action)
        }

        (res,score)
    }
    // fn solve_opt(self, mut input: Input) -> Output {
    //     let (dist_pp, par_pp) = self.dijkstra(&input);
    //     let dist = |w: &Worker, t: usize| -> u32 {
    //         if w.pos2 == !0 {
    //             dist_pp[w.pos][t]
    //         } else {
    //             if dist_pp[w.pos][t] < dist_pp[w.pos2][t] {
    //                 dist_pp[w.pos][t] + w.dist
    //             } else {
    //                 dist_pp[w.pos][t] - (dist_pp[w.pos][w.pos2] - w.dist)
    //             }
    //         }
    //     };
    //     let mut job_recv = vec![vec![];input.n_worker];
    //     // let mut res = vec![];
    //     // res.reserve(input.t_max);

    //     let mut task_done = vec![false; input.n_job];
    //     let mut cs = State::from_input(&input);
    //     for turn in 0..input.t_max {
    //         let mut add_done = vec![];
    //         for i in 0..input.n_worker {
    //             let cur_worker = &input.workers[i];
    //             let jobs_cando = input.jobs.iter().filter(|j| {
    //                 let d = dist(&cur_worker, j.v);
    //                 !task_done[j.id]
    //                     && j.n_task != 0
    //                     && cur_worker.can_do(j)
    //                     && j.deps.iter().all(|&j2| task_done[input.jobs[j2].id])
    //                     && j.can_finish(turn + d as usize, cur_worker.l_max)
    //             });
    //             let closest = jobs_cando
    //                 .map(|j| {
    //                     let d = dist(&cur_worker, j.v);
    //                     let arrive = turn as u32 + d;
    //                     let wait = j.start.saturating_sub(arrive);
    //                     (wait + d, j.id)
    //                 })
    //                 .min();
    //             if closest.is_none() {
    //                 continue;
    //             }
    //             let closest = closest.unwrap().1;
    //             let closest_job = &input.jobs[closest];
    //             job_recv[i].push(closest);
    //             if cur_worker.pos == closest_job.v && cur_worker.pos2 == !0 {
    //                 if !closest_job.can_do(turn) {
    //                     continue;
    //                 }
    //                 let task_do = input.jobs[closest].n_task.min(cur_worker.l_max);

    //                 input.jobs[closest].n_task -= task_do;
    //                 if input.jobs[closest].n_task == 0 {
    //                     add_done.push(input.jobs[closest].id);
    //                 }
    //             }
    //             // move_to_duty
    //             else {
    //                 let p = cur_worker.pos;
    //                 if p == closest_job.v {
                        
    //                 } else {
    //                 }
    //             }
    //         }
    //         // res.push(turn_action);
    //         for add in add_done {
    //             task_done[add] = true;
    //         }
    //     }
    //     Output::new(res)
    // }
}


struct Output {
    actions: Vec<Vec<Action>>
}

impl Output {
    fn new(actions: Vec<Vec<Action>>) -> Self {
        Self {
            actions
        }
    }
    fn to_string(self) -> String {
        let mut res = String::from("");
        for turn in self.actions {
            for act in turn {
                let act = match act {
                    Action::Stay => format!("stay"),
                    Action::Move(w) => format!("move {}", w + 1),
                    Action::Execute(i, a) => format!("execute {} {}", i + 1, a),
                };
                res += &act;
                res += "\n";
            }
        }
        res
    }
}

pub mod acl {
    pub struct Dsu {
        n: usize,
        parent_or_size: Vec<i32>,
    }

    impl Dsu {
        pub fn new(size: usize) -> Self {
            Self {
                n: size,
                parent_or_size: vec![-1; size],
            }
        }

        pub fn merge(&mut self, a: usize, b: usize) -> usize {
            assert!(a < self.n);
            assert!(b < self.n);
            let (mut x, mut y) = (self.leader(a), self.leader(b));
            if x == y {
                return x;
            }
            if -self.parent_or_size[x] < -self.parent_or_size[y] {
                std::mem::swap(&mut x, &mut y);
            }
            self.parent_or_size[x] += self.parent_or_size[y];
            self.parent_or_size[y] = x as i32;
            x
        }

        pub fn same(&mut self, a: usize, b: usize) -> bool {
            assert!(a < self.n);
            assert!(b < self.n);
            self.leader(a) == self.leader(b)
        }

        pub fn leader(&mut self, a: usize) -> usize {
            assert!(a < self.n);
            if self.parent_or_size[a] < 0 {
                return a;
            }
            self.parent_or_size[a] = self.leader(self.parent_or_size[a] as usize) as i32;
            self.parent_or_size[a] as usize
        }

        pub fn size(&mut self, a: usize) -> usize {
            assert!(a < self.n);
            let x = self.leader(a);
            -self.parent_or_size[x] as usize
        }

        pub fn groups(&mut self) -> Vec<Vec<usize>> {
            let mut leader_buf = vec![0; self.n];
            let mut group_size = vec![0; self.n];
            for i in 0..self.n {
                leader_buf[i] = self.leader(i);
                group_size[leader_buf[i]] += 1;
            }
            let mut result = vec![Vec::new(); self.n];
            for i in 0..self.n {
                result[i].reserve(group_size[i]);
            }
            for i in 0..self.n {
                result[leader_buf[i]].push(i);
            }
            result
                .into_iter()
                .filter(|x| !x.is_empty())
                .collect::<Vec<Vec<usize>>>()
        }
    }
}

pub fn get_time() -> f64 {
    static mut STIME: f64 = -1.0;
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap();
    let ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
    unsafe {
        if STIME < 0.0 {
            STIME = ms;
        }
        // ローカル環境とジャッジ環境の実行速度差はget_timeで吸収しておくと便利
        #[cfg(feature = "local")]
        {
            (ms - STIME) * 1.3
        }
        #[cfg(not(feature = "local"))]
        {
            ms - STIME
        }
    }
}

#[macro_use]
mod macros {
    #[macro_export]
    macro_rules! get {
        ($t:ty) => {
            {
                let mut line: String = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.trim().parse::<$t>().unwrap()
            }
        };
        ($($t:ty),*) => {
            {
                let mut line: String = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                let mut iter = line.split_whitespace();
                (
                    $(iter.next().unwrap().parse::<$t>().unwrap(),)*
                )
            }
        };
        ($t:ty; $n:expr) => {
            (0..$n).map(|_|
                get!($t)
            ).collect::<Vec<_>>()
        };
        ($($t:ty),*; $n:expr) => {
            (0..$n).map(|_|
                get!($($t),*)
            ).collect::<Vec<_>>()
        };
        ($t:ty ;;) => {
            {
                let mut line: String = String::new();
                std::io::stdin().read_line(&mut line).unwrap();
                line.split_whitespace()
                    .map(|t| t.parse::<$t>().unwrap())
                    .collect::<Vec<_>>()
            }
        };
        ($t:ty ;; $n:expr) => {
            (0..$n).map(|_| get!($t ;;)).collect::<Vec<_>>()
        };
    }
    #[allow(unused_macros)]
    #[cfg(debug_assertions)]
    #[macro_export]
    macro_rules! debug {
        ( $x: expr, $($rest:expr),* ) => {
            eprint!(concat!(stringify!($x),": {:?}, "),&($x));
            debug!($($rest),*);
        };
        ( $x: expr ) => { eprintln!(concat!(stringify!($x),": {:?}"),&($x)); };
        () => { eprintln!(); };
    }
    #[allow(unused_macros)]
    #[cfg(not(debug_assertions))]
    #[macro_export]
    macro_rules! debug {
        ( $($x: expr),* ) => {};
        () => {};
    }
    #[macro_export]
    macro_rules! chmin {
        ($base:expr, $($cmps:expr),+ $(,)*) => {{
            let cmp_min = min!($($cmps),+);
            if $base > cmp_min {
                $base = cmp_min;
                true
            } else {
                false
            }
        }};
    }
 
    #[macro_export]
    macro_rules! chmax {
        ($base:expr, $($cmps:expr),+ $(,)*) => {{
            let cmp_max = max!($($cmps),+);
            if $base < cmp_max {
                $base = cmp_max;
                true
            } else {
                false
            }
        }};
    }
 
    #[macro_export]
    macro_rules! min {
        ($a:expr $(,)*) => {{
            $a
        }};
        ($a:expr, $b:expr $(,)*) => {{
            std::cmp::min($a, $b)
        }};
        ($a:expr, $($rest:expr),+ $(,)*) => {{
            std::cmp::min($a, min!($($rest),+))
        }};
    }
    #[macro_export]
    macro_rules! max {
        ($a:expr $(,)*) => {{
            $a
        }};
        ($a:expr, $b:expr $(,)*) => {{
            std::cmp::max($a, $b)
        }};
        ($a:expr, $($rest:expr),+ $(,)*) => {{
            std::cmp::max($a, max!($($rest),+))
        }};
    }
 
    #[macro_export]
    macro_rules! mat {
        ($e:expr; $d:expr) => { vec![$e; $d] };
        ($e:expr; $d:expr $(; $ds:expr)+) => { vec![mat![$e $(; $ds)*]; $d] };
    }
}
