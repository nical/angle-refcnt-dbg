use std::{io::{self, BufRead}, collections::{HashSet, HashMap}};

struct Event {
    ref_count_before: i64,
    ref_count_after: i64,
    stack: Vec<String>,
    reverse_stack: Vec<String>,
    skip: bool,
    idx: usize,
    annotation: String,
}

fn strip_comment(line: &str) -> &str {
    if line.contains("#") {
        return line.split("#").next().unwrap_or("").trim();
    }

    line.trim()
}

fn get_arg(args: &[String], arg_name: &str) -> Option<String> {
    let mut iter = args.iter();
    let arg_name = &arg_name.to_string();
    loop {
        let arg = iter.next();
        if arg.is_none() {
            return None;
        }
        if arg == Some(arg_name) { break; }
    }

    return iter.next().cloned()
}

fn main() {
    let args = std::env::args().into_iter().collect::<Vec<_>>();
    let file_name: &str = &args[1];
    let command = if args.len() > 2 { &args[2][..] } else { &"list" };

    println!("parsing file {:?}, command {:?}", file_name, command);

    let mut skipped_frames = HashSet::new();
    iter_file_lines("skipped_frames.txt", &mut |line| {
        skipped_frames.insert(strip_comment(line).to_string());
    }).unwrap();

    let mut skipped_subtrees = HashSet::new();
    iter_file_lines("skipped_subtrees.txt", &mut |line| {
        skipped_subtrees.insert(strip_comment(line).to_string());
    }).unwrap();

    let mut renamed_frames = HashMap::new();
    iter_file_lines("renamed_frames.txt", &mut |line| {
        let mut line = strip_comment(line).split(" ");
        let from = line.next().unwrap_or("").to_string();
        let to = line.next().unwrap_or("").to_string();
        renamed_frames.insert(from, to);
    }).unwrap();

    let mut annotations = HashMap::new();

    if let Some(annotation_file_name) = get_arg(&args, "-a") {
        iter_file_lines(&annotation_file_name, &mut |line| {
            let mut line = strip_comment(line).split(": ");
            let idx = match line.next() {
                Some(idx_str) => {
                    if idx_str.is_empty() {
                        return;
                    }
                    idx_str.parse::<usize>().unwrap()
                }
                None => {
                    return;
                }
            };
            annotations.insert(idx, line.next().unwrap_or("").to_string());
        }).unwrap();    
    }

    let mut events: Vec<Event> = Vec::new();

    let mut parse_stack = false;
    let mut prev_refcnt = 2; // Seems to start at two
    let mut idx = 0;

    let file = std::fs::File::open(file_name).unwrap();
    for line in io::BufReader::new(file).lines() {
        let line = line.unwrap();
        // marker for an interesting event.
        if line.starts_with("####") || line.starts_with("--!!") {
            let mut refcnt_str = String::new();
            for c in line[40..].chars() {
                if c == ')' {
                    break;
                }
                refcnt_str.push(c);
            }

            let refcnt = refcnt_str.parse::<i64>().unwrap_or_else(|_| {
                // fall back to parsing Erich's {0x00000..something}
                let start = line.find("{0x").unwrap() + 3;
                let end = line.find("}").unwrap();
                let mut hexa_str = &line[start..end];
                while hexa_str.len() > 1 && hexa_str.chars().next() == Some('0') {
                    hexa_str = &hexa_str[1..]
                }
                i64::from_str_radix(hexa_str, 16).unwrap()
            });

            let annotation = annotations.get(&idx).unwrap_or(&String::new()).clone();

            events.push(Event {
                ref_count_before: prev_refcnt,
                ref_count_after: refcnt,
                stack: Vec::new(),
                reverse_stack: Vec::new(),
                skip: false,
                idx,
                annotation,
            });
            prev_refcnt = refcnt;
            idx += 1;

            parse_stack = true;

            continue;
        }

        if parse_stack {
            // stack frames conveniently start with a recognizable character (tab).
            if line.starts_with("\t") {
                let mut frame = line[1..].to_string();
                if let Some(name) = renamed_frames.get(&frame) {
                    frame = name.clone();
                }

                if frame.len() > 2 && !skipped_frames.contains(&frame) {
                    let event = &mut events.last_mut().unwrap();

                    // mark the event if the stack frame is in the skip list for sub-trees.
                    if skipped_subtrees.contains(&frame) {
                        event.skip = true;
                    }

                    // Skip over recursions.
                    if event.stack.last() != Some(&frame) {
                        event.stack.push(frame);
                    }
                }
            } else {
                // Stop as soon as there is a line that does not start with a tab
                // to avoid picking up som log.
                parse_stack = false;
            }
        }
    }

    //events.retain(|e| !e.skip);

    for event in &mut events {
        event.reverse_stack = event.stack.iter().rev().cloned().collect();
    }

    match command {
        "list" => {
            print_events(&events, &args[3..]);
        }
        "tree" => {
            print_tree(&events, &args[3..]);
        }
        "count" => {
            count_events(&events, &args[3..])
        }
        _ => {
            println!("invalid command {:?}", command);
        }
    }
}

fn event_symbol(event: &Event) -> &'static str {
    match event.ref_count_after - event.ref_count_before {
        1 => "++",
        -1 => "--",
        _ => "!!",
    }
}

fn print_events(events: &[Event], args: &[String]) {
    let stack_depth = if args.len() > 0 {
        args[0].parse::<usize>().unwrap()
    } else {
        0
    };

    for event in events {
        let symbol = event_symbol(event);
        if event.skip {
            println!("[skipped {}]", symbol);
            continue;
        }
        println!("{} ref {} -> {:?}", symbol, event.ref_count_before, event.ref_count_after);
        for i in 0..stack_depth {
            if event.stack.len() <= i {
                break;
            }
            println!("    {}", event.stack[i]);
        }
    }
}

fn indent(n: usize) {
    for _ in 0..n {
        print!("| ");
    }
}

fn print_tree(events: &[Event], args: &[String]) {
    let filter = get_arg(args, "-f");

    let mut prev_stack: &[String] = &[];
    for event in events {
        let symbol = event_symbol(event);

        let mut filtered_out = false;
        if let Some(pattern) = &filter {
            filtered_out = true;
            for frame in &event.stack {
                if frame.contains(&pattern[..]) {
                    filtered_out = false;
                    break;
                }
            }
        }

        if filtered_out {
            continue;
        }

        if event.skip {
            println!("[skipped {}] {}", symbol, event.annotation);
            continue;
        }

        let stack = &event.reverse_stack[..];
        let common_depth: usize = prev_stack.iter().zip(stack.iter()).fold(0, |count, items| {
            if items.0 == items.1 { count + 1 } else { count }
        });

        let mut tabs = common_depth;
        for frame in &stack[common_depth..] {
            indent(tabs);
            println!("{}", frame);
            tabs += 1;
        }

        indent(tabs);

        println!("{} ref {} -> {:?}    (#{}) {}", symbol, event.ref_count_before, event.ref_count_after, event.idx, event.annotation);

        prev_stack = stack;
    }
}

fn count_events(events: &[Event], args: &[String]) {
    let mut n = 0;
    let mut inc = 0;
    let mut dec = 0;

    let pattern = &args[0];

    for event in events {
        for frame in &event.stack {
            if frame.contains(&pattern[..]) {
                let symbol = event_symbol(event);

                if event.skip {
                    println!("[skipped] {} ref {} -> {:?}    (#{}) {}", symbol, event.ref_count_before, event.ref_count_after, event.idx, event.annotation);
                } else {
                    println!("{} ref {} -> {:?}    (#{}) {}", symbol, event.ref_count_before, event.ref_count_after, event.idx, event.annotation);
                }

                n += 1;
                match symbol {
                    "++" => { inc += 1; }
                    "--" => { dec += 1; }
                    _ => {}
                }
                break;
            }
        }
    }

    println!("Found {} events, ++{}/--{}", n, inc, dec);
}


fn iter_file_lines(file_name: &str, cb: &mut dyn FnMut(&str)) -> Result<(), std::io::Error> {
    let file = std::fs::File::open(file_name)?;

    for line in io::BufReader::new(file).lines() {
        let line = line.unwrap();
        cb(&line);
    }

    Ok(())
}
