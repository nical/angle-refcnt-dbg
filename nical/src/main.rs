use std::io::{self, BufRead};

struct Event {
    ref_count_before: i64,
    ref_count_after: Option<i64>,
    stack: Vec<String>,
    reverse_stack: Vec<String>,
}

fn main() {
    let args = std::env::args().into_iter().collect::<Vec<_>>();
    let file_name: &str = &args[1];
    let command = if args.len() > 2 { &args[2][..] } else { &"list" };

    println!("parsing file {:?}, command {:?}", file_name, command);

    let file = std::fs::File::open(file_name).unwrap();

    let mut events: Vec<Event> = Vec::new();

    let mut parse_stack = false;

    for line in io::BufReader::new(file).lines() {
        let line = line.unwrap();
        if line.starts_with("####") {
            let mut refcnt_str = String::new();
            for c in line[40..].chars() {
                if c == ')' {
                    break;
                }
                refcnt_str.push(c);
            }

            let refcnt = refcnt_str.parse::<i64>().unwrap();

            if let Some(prev) = events.last_mut() {
                prev.ref_count_after = Some(refcnt);
            }

            events.push(Event {
                ref_count_before: refcnt,
                ref_count_after: None,
                stack: Vec::new(),
                reverse_stack: Vec::new(),
            });

            parse_stack = true;

            continue;
        }

        if parse_stack {
            if line.starts_with("\t") {
                events.last_mut().unwrap().stack.push(line[1..].to_string());
            } else {
                parse_stack = false;
            }
        }
    }

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
        _ => {
            println!("invalid command {:?}", command);
        }
    }
}

fn event_symbol(event: &Event) -> &'static str {
    let ref_count_after = event.ref_count_after.unwrap_or(1234567);
    match ref_count_after - event.ref_count_before {
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
        let ref_count_after = event.ref_count_after.unwrap_or(1234567);
        let symbol = event_symbol(event);
        println!("{} ref {} -> {:?}", symbol, event.ref_count_before, ref_count_after);
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
    let mut prev_stack: &[String] = &[];
    for event in events {
        let stack = &event.reverse_stack[..];
        let common_depth: usize = prev_stack.iter().zip(stack.iter()).fold(0, |count, items| {
            if items.0 == items.1 { count + 1 } else { count }
        });

        let mut tabs = common_depth;
        for frame in &event.stack[common_depth..] {
            indent(tabs);
            println!("{:?}", frame);
            tabs += 1;
        }

        indent(tabs);
        let symbol = event_symbol(event);

        println!("{} ref {} -> {:?}", symbol, event.ref_count_before, event.ref_count_after);

        prev_stack = stack;
    }
}
