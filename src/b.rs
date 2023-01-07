use proconio::{input, marker::Usize1, source::line::LineSource};
use std::io::{BufReader, Stdin};

fn main() {
    let mut stdin =
        proconio::source::line::LineSource::new(std::io::BufReader::new(std::io::stdin()));
    let input = Input::from_stdin(&mut stdin);
    let solver = Solver::new();
    let output = solver.solve(input);
    for turn in output {
        for act in turn {
            let act = match act {
                Action::Stay => format!("stay"),
                Action::Move(w) => format!("move {}", w + 1),
                Action::Execute(i, a) => format!("execute {} {}", i + 1, a),
            };
            println!("{}", act);
        }
    }
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
    reward: Vec<(u32, u64)>,
    deps: Vec<usize>,
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
        reward: Vec<(u32, u64)>,
        deps: Vec<usize>,
    ) -> Self {
        Self {
            id,
            job_type,
            n_task,
            v,
            reward,
            deps,
            penalty,
            w_depend,
            mandatory,
        }
    }
    fn can_finish(&self, start_turn: usize, l_max: u32) -> bool {
        let end_turn = start_turn as u32 + (self.n_task + l_max - 1) as u32 / l_max;
        end_turn < self.reward[self.reward.len() - 1].0
    }
    fn can_do(&self, turn: usize) -> bool {
        self.reward[0].0 < (turn as u32) && (turn as u32) < self.reward[self.reward.len() - 1].0
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
    fn next_states(&self, input: &Input) -> Vec<(Vec<Action>, State)> {
        // let mut res = vec![];

        vec![]
    }
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
    fn solve(self, mut input: Input) -> Vec<Vec<Action>> {
        let (dist_pp, par_pp) = self.dijkstra(&input);
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
                        let arrive = turn as u32 + d;
                        let wait = j.reward[0].0.saturating_sub(arrive);
                        (wait + d, j.id)
                    })
                    .min();
                if closest.is_none() {
                    turn_action.push(Action::Stay);
                    continue;
                }
                // eprintln!("dist: {:?}",closest.unwrap());
                let closest = closest.unwrap().1;
                let closest_job = &input.jobs[closest];
                // eprintln!("v: {},id: {}",closest_job.v,closest_job.id);
                // do_work
                // if dist_pp[closest_job.v][cur_worker.pos] == 0 {
                if cur_worker.pos == closest_job.v && cur_worker.pos2 == !0 {
                    if !closest_job.can_do(turn) {
                        turn_action.push(Action::Stay);
                        continue;
                    }
                    let task_do = input.jobs[closest].n_task.min(cur_worker.l_max);
                    turn_action.push(Action::Execute(closest_job.id, task_do));
                    // turn_action.push(Action::Stay);

                    input.jobs[closest].n_task -= task_do;
                    if input.jobs[closest].n_task == 0 {
                        // task_done[input.jobs[closest].id] = true;
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
        res
    }
    fn solve_beam_search(self, mut input: Input) -> Vec<Vec<Action>> {
        vec![]
    }
}
