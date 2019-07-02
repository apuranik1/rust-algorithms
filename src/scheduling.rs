pub struct Interval<T: Ord> {
    pub start: T,
    pub end: T,
}

impl<T: Ord> Interval<T> {
    pub fn new(start: T, end: T) -> Self {
        Interval { start, end }
    }
}

pub struct WeightedInterval<T: Ord> {
    pub start: T,
    pub end: T,
    pub weight: i32,
}

impl<T: Ord> WeightedInterval<T> {
    pub fn new(start: T, end: T, weight: i32) -> Self {
        WeightedInterval { start, end, weight }
    }
}

pub fn schedule_intervals<T: Ord>(intervals: &[Interval<T>]) -> Vec<usize> {
    if intervals.is_empty() {
        Vec::new()
    } else {
        let mut sorted_by_end: Vec<usize> = (0..intervals.len()).collect();
        sorted_by_end.sort_unstable_by_key(|i| &intervals[*i].end);
        let mut scheduled = vec![sorted_by_end[0]];
        for index in sorted_by_end {
            if intervals[index].start >= intervals[*scheduled.last().unwrap()].end {
                scheduled.push(index);
            }
        }
        scheduled
    }
}

pub fn schedule_weighted_intervals<T: Ord>(intervals: &[WeightedInterval<T>]) -> Vec<usize> {
    use std::cmp::Ordering::*;
    if intervals.is_empty() {
        return Vec::new();
    }
    let mut sorted_by_end: Vec<usize> = (0..intervals.len()).collect();
    sorted_by_end.sort_unstable_by_key(|i| &intervals[*i].end);
    // best result using only intervals ending <= i
    let mut best_until = Vec::new();
    // the last scheduled interval to achieve the above
    let mut last_used: Vec<usize> = Vec::new();
    // if last_used[i] == i, then the previous interval if one exists
    let mut prev: Vec<Option<usize>> = Vec::new();
    best_until.push(intervals[sorted_by_end[0]].weight);
    last_used.push(0);
    prev.push(None);

    for (i, int_index) in sorted_by_end.iter().cloned().enumerate().skip(1) {
        // opt when we don't use current interval
        let previous_opt = best_until[i - 1];
        // current interval
        let current = &(intervals[int_index]);
        // slice up to i so that we only consider previously seen intervals
        // otherwise 0-length intervals can have strange outcomes
        let previous_schedulable = sorted_by_end[..i].binary_search_by(|index: &usize| {
            if intervals[*index].end <= current.start {
                Less
            } else {
                Greater
            }
        }); // never returns Equal, so we get one past the last schedulable interval
        let previous_schedulable: usize = previous_schedulable.unwrap_err();
        let opt_with_current = match previous_schedulable {
            0 => {
                prev.push(None);
                current.weight
            }
            p => {
                prev.push(Some(last_used[p - 1])); // use optimum from p - 1
                current.weight + best_until[p - 1]
            }
        };
        if previous_opt > opt_with_current {
            best_until.push(previous_opt);
            last_used.push(last_used[i - 1]);
        } else {
            best_until.push(opt_with_current);
            last_used.push(i);
        }
    }
    let mut scheduled = Vec::new();
    // collect actually scheduled intervals
    let mut next_scheduled = *last_used.last().unwrap();
    loop {
        scheduled.push(sorted_by_end[next_scheduled]);
        match prev[next_scheduled] {
            Some(p) => next_scheduled = p,
            None => break,
        }
    }
    scheduled.reverse();
    scheduled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduling_1() {
        let intervals = vec![
            Interval::new(3, 5),
            Interval::new(1, 3),
            Interval::new(2, 4),
            Interval::new(0, 2),
            Interval::new(0, 1),
        ];
        let expected = vec![4, 1, 0];
        assert_eq!(schedule_intervals(&intervals), expected);
    }

    #[test]
    fn test_weighted_scheduling_1() {
        let intervals = vec![
            WeightedInterval::new(3, 5, 1),
            WeightedInterval::new(1, 3, 1),
            WeightedInterval::new(2, 4, 1),
            WeightedInterval::new(0, 2, 1),
            WeightedInterval::new(0, 1, 1),
        ];
        let expected = vec![4, 1, 0];
        assert_eq!(schedule_weighted_intervals(&intervals), expected);
    }

    #[test]
    fn test_weighted_scheduling_2() {
        let intervals = vec![
            WeightedInterval::new(1, 5, 2),
            WeightedInterval::new(1, 3, 1),
            WeightedInterval::new(2, 4, 1),
            WeightedInterval::new(0, 2, 3),
            WeightedInterval::new(0, 1, 1),
        ];
        let expected = vec![3, 2];
        assert_eq!(schedule_weighted_intervals(&intervals), expected);
    }
}
