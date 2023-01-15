use proconio::{input, marker::Usize1, source::line::LineSource};
use rand::prelude::*;
use std::collections::HashMap;
use std::io::{BufReader, Stdin};
use std::{cmp::Reverse, collections::BinaryHeap, collections::VecDeque};

fn main() {
    get_time();
    let mut stdin =
        proconio::source::line::LineSource::new(std::io::BufReader::new(std::io::stdin()));
    let input = Input::from_stdin(&mut stdin);
    // eprintln!("{:?}",get_time());
    let solver = Solver::new(&input);
    let output = solver.solve(input);
    // eprintln!("{:?}",get_time());
    print!("{}", output.to_string());
    input! {
        from &mut stdin,
        score: u64
    }
    println!("{}", score);
    // eprintln!("{:?}",get_time());
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
        let start = raw_reward[0].0 as usize - 1;
        let end = raw_reward[raw_reward.len() - 1].0 as usize - 1;
        let mut reward = vec![];
        for i in 0..raw_reward.len() - 1 {
            for t in raw_reward[i].0..raw_reward[i + 1].0 {
                if t == 0 {
                    continue;
                }
                let y_prev = raw_reward[i].1;
                let y_next = raw_reward[i + 1].1;
                let t_prev = raw_reward[i].0 - 1;
                let t_next = raw_reward[i + 1].0 - 1;
                reward.push(
                    (y_next - y_prev) * (t - t_prev) as u64 / (t_next - t_prev) as u64 + y_prev,
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
    fn get_reward(&self, t: usize) -> u64 {
        if t <= self.start || self.end <= t {
            0
        } else {
            self.reward[t - self.start - 1]
        }
    }
    fn can_finish(&self, start_turn: usize, l_max: u32) -> bool {
        let end_turn = start_turn + ((self.n_task + l_max - 1) / l_max as u32) as usize;
        end_turn < self.end
    }
    fn can_do(&self, turn: usize) -> bool {
        self.start < turn && turn < self.end
    }
}

#[derive(Debug, Clone, Copy)]
enum Action {
    Stay,
    Move(usize),
    Execute(usize, u32),
}

// inputで状態を表すもののみ保持
#[derive(Debug, Clone)]
struct State {
    turn: usize,
    worker_pos: Vec<(usize, usize, u32)>,
    job_remain: Vec<u32>,
    reward_got: Vec<u64>,
}

impl State {
    fn new(
        turn: usize,
        worker_pos: Vec<(usize, usize, u32)>,
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
            worker_pos.push((
                input.workers[i].pos,
                input.workers[i].pos2,
                input.workers[i].dist,
            ));
        }
        let mut job_remain = vec![];
        for i in 0..input.n_job {
            job_remain.push(input.jobs[i].n_task);
        }
        let reward_got = vec![0; input.n_job];
        Self {
            turn: 0,
            worker_pos,
            job_remain,
            reward_got,
        }
    }
    fn next_actions(&self,input: &Input,pos_work: &Vec<Option<usize>>,num: usize) -> Vec<(Vec<Action>,u64)> {
        let mut each_acts = vec![vec![];input.n_worker];
        // 各ワーカーの良い行動を2つ取ってその行動の直積を取る
        for i in 0..input.n_worker {
            let cp_score = self.pos_score(input, pos_work, self.worker_pos[i].0);
            if self.worker_pos[i].1 != !0 {
                each_acts[i] = vec![(0,Action::Move(self.worker_pos[i].1))];
                continue;
            }
            let mut cur = vec![];
            if let Some(j) = pos_work[self.worker_pos[i].0] {
                if input.workers[i].can_do(&input.jobs[j])
                && input.jobs[j].deps.iter().all(|&x| self.job_remain[x] == 0)
                && input.jobs[j].can_do(self.turn) {
                    let a = input.workers[i].l_max.min(self.job_remain[i]);
                    let execute_score =
                        ((a as u64 * input.jobs[j].get_reward(self.turn)) as f64
                        * (-(self.job_remain[j] as f64)).exp()) as u64;
                        // * (input.jobs[j].n_task as u64 - self.job_remain[j] as u64) / input.jobs[j].n_task as u64;
                    cur.push((cp_score + execute_score,Action::Execute(j, a)));
                }
            }
            for &(to,cost) in &input.graph[self.worker_pos[i].0] {
                let tp_score = self.pos_score(input, pos_work, to);
                cur.push(((cp_score * (cost - 1) as u64 + tp_score) / cost as u64,Action::Move(to)));
            }
            cur.push((cp_score,Action::Stay));
            if cur.len() > 2 {
                cur.select_nth_unstable_by_key(2, |&x| Reverse(x.0));
                cur.truncate(2);
            }
            each_acts[i] = cur;
        }
        let mut res: Vec<(Vec<Action>, u64)> = vec![(vec![],0)];
        for i in 0..input.n_worker {
            let mut cur = vec![];
            for (cur_acts,cur_score) in &res {
                for &(add,act) in &each_acts[i] {
                    let mut nx = cur_acts.clone();
                    nx.push(act);
                    cur.push((nx,cur_score + add));
                }
            }
            if cur.len() > num && i != input.n_worker - 1 {
                cur.select_nth_unstable_by_key(num, |x| Reverse(x.1));
                cur.truncate(num);
            }
            res = cur;
        }
        // 同じ仕事をやりすぎていないか
        for i in 0..res.len() {
            let mut cs = self.clone();
            for j in 0..input.n_worker {
                if let Action::Execute(idx, a) = res[i].0[j] {
                    if cs.job_remain[idx] < a {
                        if cs.job_remain[idx] == 0 {
                            res[i].0[j] = Action::Stay;
                        }
                        else {
                            res[i].0[j] = Action::Execute(idx, cs.job_remain[idx]);
                        }
                    }
                    let dummy = vec![];
                    cs.apply_action(input, &dummy, idx, res[i].0[j]);
                }
            }
        }
        res.sort_by_key(|x| Reverse(x.1));
        res
    }
    // 現時点でのpのスコア
    fn pos_score(&self,input: &Input,pos_work: &Vec<Option<usize>>,p: usize) -> u64 {
        let mut res = 0;
        let mut que = BinaryHeap::new();
        que.push((0,p));
        let mut dist: HashMap<usize, u64> = HashMap::new();
        dist.insert(p, 0);
        while let Some((d,x)) = que.pop() {
            if dist.contains_key(&x) && *dist.get_key_value(&x).unwrap().1 < d {
                continue;
            }
            if d > 10 {
                break
            }
            if let Some(id) = pos_work[x] {
                let cj = &input.jobs[id];
                if self.turn + d as usize <= cj.start {
                    res += cj.get_reward(cj.start + 1) / 2 / (d + 1) as u64;
                }
                else {
                    res += cj.get_reward(cj.start + 1) / (d + 1) as u64;
                }
            }
            for &(to,cost) in &input.graph[x] {
                if !dist.contains_key(&to) || (d + cost as u64) < *dist.get_key_value(&to).unwrap().1 {
                    dist.insert(to, d + cost as u64);
                    que.push((d + cost as u64,to));
                }
            }
        }
        res
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
    fn apply_action(&mut self, input: &Input, dist_pp: &Vec<Vec<u32>>, idx: usize, action: Action) {
        match action {
            Action::Stay => {}
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
                if self.worker_pos[idx].2 == dist_pp[self.worker_pos[idx].0][self.worker_pos[idx].1]
                {
                    self.worker_pos[idx].0 = self.worker_pos[idx].1;
                    self.worker_pos[idx].1 = !0;
                    self.worker_pos[idx].2 = 0;
                } else if self.worker_pos[idx].2 == 0 {
                    self.worker_pos[idx].1 = !0;
                }
            }
            Action::Execute(i, a) => {
                self.job_remain[i] -= a;
                self.reward_got[i] += input.jobs[i].get_reward(self.turn) * a as u64;
            }
        };
    }
    fn tick(&mut self) {
        self.turn += 1;
    }
    fn try_apply(&self,op:usize,score:usize,hash:u64) -> (usize,u64) {
        todo!()
    }
    fn apply(&mut self, input: &Input, dist_pp: &Vec<Vec<u32>>,actions: &Vec<Action>) {
        for i in 0..input.n_worker {
            self.apply_action(input, dist_pp, i, actions[i]);
        }
        self.tick();
    }
    fn back(&mut self,backup: u128){
        todo!()
    }
    fn hash(&self)->u64{
        todo!()
    }
}

struct Solver {
    dist_pp: Vec<Vec<u32>>,
    par_pp: Vec<Vec<usize>>,
    pos_work: Vec<Option<usize>>,
}

impl Solver {
    fn new(input: &Input) -> Self {
        let (dist_pp,par_pp) = Solver::dijkstra(input);
        let mut pos_work = vec![None;input.v];
        for i in 0..input.n_job {
            pos_work[input.jobs[i].v] = Some(i);
        }
        Self {
            dist_pp,par_pp,pos_work
        }
    }
    fn dijkstra(input: &Input) -> (Vec<Vec<u32>>, Vec<Vec<usize>>) {
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
    fn solve(self, input: Input) -> Output {
        eprintln!("{:?}", get_time());
        let mut cs = State::from_input(&input);
        let dist = |p: (usize,usize,u32), to: usize| -> u32 {
            if p.1 == !0 {
                self.dist_pp[p.0][to]
            } else {
                if self.dist_pp[p.0][to] < self.dist_pp[p.1][to] {
                    self.dist_pp[p.0][to] + p.2
                } else {
                    self.dist_pp[p.0][to] - (self.dist_pp[p.0][p.1] - p.2)
                }
            }
        };
        let mut res = vec![];
        res.reserve(input.t_max);

        let mut task_done = vec![false; input.n_job];
        let mut task_do = vec![!0; input.n_worker];
        for turn in 0..input.t_max {
            let mut turn_action = vec![];
            turn_action.reserve(input.t_max);
            let mut add_done = vec![];
            for i in 0..input.n_worker {
                if task_do[i] == !0 || cs.job_remain[task_do[i]] == 0 {
                    let jobs_cando = (0..input.n_job)
                        .map(|jid| (jid,dist(cs.worker_pos[i],input.jobs[jid].v)))
                        .filter(|&(jid,d)| {
                        !task_done[jid]
                        && cs.job_remain[jid] != 0
                        && input.workers[i].can_do(&input.jobs[jid])
                        && input.jobs[jid].deps.iter().all(|&j2| task_done[input.jobs[j2].id])
                        && input.jobs[jid].can_finish(turn + d as usize, input.workers[i].l_max)
                        && task_do.iter().all(|&jid2| jid2 != jid)
                    });
                    let closest = jobs_cando.min_by_key(|&(jid,d)| {
                        let arrive = turn + d as usize;
                        let wait = input.jobs[jid].start.saturating_sub(arrive);
                        wait + d as usize
                    });
                    if closest.is_none() {
                        turn_action.push(Action::Stay);
                        continue;
                    }
                    let closest = closest.unwrap().0;
                    task_do[i] = closest;
                }
                // do_work
                if cs.worker_pos[i].0 == input.jobs[task_do[i]].v && cs.worker_pos[i].1 == !0 {
                    if !input.jobs[task_do[i]].can_do(turn) {
                        turn_action.push(Action::Stay);
                        continue;
                    }
                    let task_amount = cs.job_remain[task_do[i]].min(input.workers[i].l_max);
                    turn_action.push(Action::Execute(task_do[i], task_amount));
                    cs.apply_action(&input, &self.dist_pp, i, Action::Execute(task_do[i], task_amount));

                    if cs.job_remain[task_do[i]] == 0 {
                        add_done.push(input.jobs[task_do[i]].id);
                        task_do[i] = !0;
                    }
                }
                // move_to_duty
                else {
                    let p = cs.worker_pos[i];
                    let jp = input.jobs[task_do[i]].v;
                    if p.0 == jp {
                        turn_action.push(Action::Move(p.0));
                        cs.apply_action(&input, &self.dist_pp, i, Action::Move(p.0));
                    } else {
                        turn_action.push(Action::Move(self.par_pp[jp][p.0]));
                        cs.apply_action(&input, &self.dist_pp, i, Action::Move(self.par_pp[jp][p.0]));
                    }
                }
            }
            res.push(turn_action);
            for add in add_done {
                task_done[add] = true;
            }
            cs.tick();
        }
        Output::new(res)
    }
}



struct Output {
    actions: Vec<Vec<Action>>,
}

impl Output {
    fn new(actions: Vec<Vec<Action>>) -> Self {
        Self { actions }
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
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
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
