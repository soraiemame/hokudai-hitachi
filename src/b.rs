use proconio::{input, marker::Usize1, source::line::LineSource};
use std::io::{BufReader, Stdin};
use std::{cmp::Reverse, collections::BinaryHeap};

fn main() {
    let mut stdin =
        proconio::source::line::LineSource::new(std::io::BufReader::new(std::io::stdin()));
    let input = Input::from_stdin(&mut stdin);
    let solver = Solver::new(&input);
    solver.solve(&input);
    input! {
        from &mut stdin,
        score: u64
    }
    println!("{}", score);
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
    t_weather: usize,
    n_weather: usize,
    prob_weather: Vec<Vec<f64>>,
    w_constraint: Vec<u32>,
    p_m: f64,
    r_m: f64,
    alpha: f64,
    w_prob: Vec<(usize, Vec<f64>)>,
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
                penalty: f64,
                w_depend: f64,
                mandatory: i32,
                reward: [(u32,u64)],
                deps: [Usize1],
            }
            jobs.push(Job::new(
                id,
                job_type,
                n_task,
                v,
                penalty,
                w_depend,
                mandatory == 1,
                reward,
                deps,
            ));
        }
        input! {
            from &mut stdin,
            t_weather: usize,
            n_weather: usize,
            prob_weather: [[f64;n_weather];n_weather],
            w_constraint: [u32;n_weather],
            p_m: f64,
            r_m: f64,
            alpha: f64,
        }
        let mut w_prob = vec![];
        w_prob.reserve(t_max / t_weather);
        for _ in 0..t_max / t_weather {
            input! {
                from &mut stdin,
                t: usize,
                prob: [f64;n_weather]
            }
            w_prob.push((t, prob));
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
            t_weather,
            n_weather,
            prob_weather,
            w_constraint,
            p_m,
            r_m,
            alpha,
            w_prob,
        }
    }
}

// return weather
fn do_turn_input(
    input: &Input,
    turn: usize,
    mut stdin: &mut LineSource<BufReader<Stdin>>,
) -> (usize, Option<Vec<(usize, Vec<f64>)>>) {
    input! {
        from &mut stdin,
        w: usize,
        _jobs_selected: [(usize,u32)],
        _workers: [(usize,usize,usize,u32);input.n_worker]
    }
    if turn % input.t_weather == 0 {
        let mut w_prob = vec![];
        w_prob.reserve((input.t_max - turn) / input.t_weather);
        for _ in 0..(input.t_max - turn) / input.t_weather {
            input! {
                from &mut stdin,
                t: usize,
                prob: [f64;input.n_weather]
            }
            w_prob.push((t, prob));
        }
        (w, Some(w_prob))
    } else {
        (w, None)
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
    penalty: f64,
    w_depend: f64,
    mandatory: bool,
    // (start,end)
    start: usize,
    end: usize,
    // reward: Vec<(u32, u64)>,
    reward: Vec<u64>,
    deps: Vec<usize>,
    max_reward: u64,
}

impl Job {
    fn new(
        id: usize,
        job_type: usize,
        n_task: u32,
        v: usize,
        penalty: f64,
        w_depend: f64,
        mandatory: bool,
        raw_reward: Vec<(u32, u64)>,
        deps: Vec<usize>,
    ) -> Self {
        let start = raw_reward[0].0 as usize - 1;
        let end = raw_reward[raw_reward.len() - 1].0 as usize - 1;
        let mut reward = vec![];
        let mut max_reward = 0;
        for i in 0..raw_reward.len() - 1 {
            for t in raw_reward[i].0..raw_reward[i + 1].0 {
                let y_prev = raw_reward[i].1 as i64;
                let y_next = raw_reward[i + 1].1 as i64;
                let t_prev = raw_reward[i].0;
                let t_next = raw_reward[i + 1].0;
                chmax!(max_reward, y_prev as u64);
                reward.push(
                    ((y_next - y_prev) * (t - t_prev) as i64 / (t_next - t_prev) as i64 + y_prev)
                        as u64,
                );
            }
        }
        Self {
            id,
            job_type,
            n_task,
            v,
            penalty,
            w_depend,
            mandatory,
            start,
            end,
            reward,
            deps,
            max_reward,
        }
    }
    fn get_reward(&self, t: usize) -> u64 {
        if t <= self.start || self.end <= t {
            0
        } else {
            self.reward[t - self.start]
        }
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
    worker_pos: Vec<usize>,
    job_remain: Vec<u32>,
    reward_got: Vec<u64>,
}

impl State {
    fn new(
        turn: usize,
        worker_pos: Vec<usize>,
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
    
}

struct Solver {
    dist_pp: Vec<Vec<u32>>,
    ord: Vec<Vec<usize>>,
    par_pp: Vec<Vec<usize>>,
    pos_work: Vec<Vec<usize>>,
}

impl Solver {
    fn new(input: &Input) -> Self {
        let (dist_pp, par_pp) = Solver::dijkstra(input);
        let mut ord = vec![];
        ord.reserve(input.v);
        for i in 0..input.v {
            let mut v = (0..input.v).collect::<Vec<_>>();
            v.sort_by_key(|&x| dist_pp[i][x]);
            ord.push(v);
        }
        let mut pos_work = vec![vec![]; input.v];
        for i in 0..input.n_job {
            pos_work[input.jobs[i].v].push(i);
        }
        Self {
            dist_pp,
            ord,
            par_pp,
            pos_work,
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
    fn dist(&self, p: (usize, usize, u32), to: usize) -> u32 {
        if p.1 == !0 {
            self.dist_pp[p.0][to]
        } else {
            if self.dist_pp[p.0][to] < self.dist_pp[p.1][to] {
                self.dist_pp[p.0][to] + p.2
            } else {
                self.dist_pp[p.0][to] - (self.dist_pp[p.0][p.1] - p.2)
            }
        }
    }
    fn solve(&self,input: &Input) {
        todo!();
    }
    fn get_jobs_around(
        &self,
        input: &Input,
        cs: &State,
        task_do: &Vec<usize>,
        wid: usize,
    ) -> Option<usize> {
        todo!()
    }
    fn run(&self, input: &Input, cs: &mut State) -> Vec<Vec<Action>> {
        todo!()
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

