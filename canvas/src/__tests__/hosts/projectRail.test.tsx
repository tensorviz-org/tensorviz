import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';

import { ProjectRail, experimentMatches } from '../../ide/ProjectRail';
import { ProjectIndexSummary } from '../../ide/localBrowserState';

function experiment(overrides: Partial<ProjectIndexSummary['experiments'][0]> = {}) {
  return {
    experiment_id: 'exp:1',
    label: 'train',
    entrypoint: 'train.py',
    root: '',
    status: { state: 'active' as const, declared: false },
    closure: ['train.py', 'models/net.py'],
    ...overrides,
  };
}

function index(overrides: Partial<ProjectIndexSummary> = {}): ProjectIndexSummary {
  return {
    experiments: [experiment()],
    unreached: [],
    roots: [''],
    truncated: false,
    ...overrides,
  };
}

const noop = () => undefined;

describe('ProjectRail — what it refuses to hide', () => {
  it('marks a candidate as needing confirmation rather than asserting it', () => {
    // A filename is the weakest evidence there is. Presenting it as an
    // experiment would claim more than the code proved.
    render(
      <ProjectRail
        index={index({
          experiments: [
            experiment({ experiment_id: 'a', label: 'sweep', status: { state: 'candidate', declared: false } }),
          ],
        })}
        focusedId=""
        onFocus={noop}
      />,
    );
    expect(screen.getByText('candidate')).toBeInTheDocument();
  });

  it('says when a file will not parse, instead of leaving a hole', () => {
    render(
      <ProjectRail
        index={index({ unreached: [{ file: 'broken.py', error: 'invalid syntax (line 2)' }] })}
        focusedId=""
        onFocus={noop}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /no experiment reaches/i }));
    expect(screen.getByText(/invalid syntax \(line 2\)/)).toBeInTheDocument();
  });

  it('admits a truncated index rather than looking complete', () => {
    // Two numbers rather than one flag (step 3.5). "Part of the project" is not
    // something a reader can act on: 592 of 596 is a rounding error, and 200 of
    // 4,000 is a different tool.
    render(
      <ProjectRail
        index={index({ truncated: true })}
        focusedId=""
        onFocus={noop}
        indexed={200}
        discovered={5000}
      />,
    );
    expect(screen.getByRole('status')).toHaveTextContent('indexed 200 of 5,000 files');
  });

  it('still admits truncation when it cannot say how much', () => {
    // The counts ride on the message beside the index, and an older host does not
    // send them. Saying nothing at all would make a capped project look whole.
    render(<ProjectRail index={index({ truncated: true })} focusedId="" onFocus={noop} />);
    expect(screen.getByRole('status')).toHaveTextContent('indexed part of the project');
  });

  it('says nothing about counts when the index is complete', () => {
    render(
      <ProjectRail index={index()} focusedId="" onFocus={noop} indexed={12} discovered={12} />,
    );
    expect(screen.queryByText(/indexed/)).toBeNull();
  });

  it('tells the user what would make a file an experiment when none exist', () => {
    render(<ProjectRail index={index({ experiments: [] })} focusedId="" onFocus={noop} />);
    expect(screen.getByText(/tensorviz.experiment\(\)/)).toBeInTheDocument();
  });
});

describe('ProjectRail — grouping and focus', () => {
  it('does not group a flat repository', () => {
    render(<ProjectRail index={index()} focusedId="" onFocus={noop} />);
    expect(screen.queryByText('project root')).not.toBeInTheDocument();
  });

  it('groups by derived root when there is more than one project', () => {
    render(
      <ProjectRail
        index={index({
          experiments: [
            experiment({ experiment_id: 'a', root: 'packages/alpha' }),
            experiment({ experiment_id: 'b', root: 'packages/beta', label: 'other' }),
          ],
          roots: ['packages/alpha', 'packages/beta'],
        })}
        focusedId=""
        onFocus={noop}
      />,
    );
    // By role: since the project picker exists, each root also appears as an
    // <option>, and a bare getByText would match two nodes and fail for a reason
    // that has nothing to do with grouping.
    expect(screen.getByRole('heading', { name: 'packages/alpha' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'packages/beta' })).toBeInTheDocument();
  });

  it('offers a project picker only when the workspace holds more than one', () => {
    // The SCM view's repository picker, one level above the experiments. A flat
    // repository must not be made to look like a monorepo, so a single root gets
    // no control at all.
    const { rerender } = render(
      <ProjectRail index={index()} focusedId="" onFocus={noop} />,
    );
    expect(screen.queryByLabelText('Which project to show')).not.toBeInTheDocument();

    rerender(
      <ProjectRail
        index={index({
          experiments: [
            experiment({ experiment_id: 'a', root: 'packages/alpha' }),
            experiment({ experiment_id: 'b', root: 'packages/beta', label: 'other' }),
          ],
          roots: ['packages/alpha', 'packages/beta'],
        })}
        focusedId=""
        onFocus={noop}
      />,
    );
    expect(screen.getByLabelText('Which project to show')).toBeInTheDocument();
  });

  it('narrows to one project, and says that all of them is the default', () => {
    render(
      <ProjectRail
        index={index({
          experiments: [
            experiment({ experiment_id: 'a', root: 'packages/alpha' }),
            experiment({ experiment_id: 'b', root: 'packages/beta', label: 'other' }),
          ],
          roots: ['packages/alpha', 'packages/beta'],
        })}
        focusedId=""
        onFocus={noop}
      />,
    );
    // A workspace that was never narrowed has not been narrowed.
    expect(screen.getByText('All 2 projects')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'packages/beta' })).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Which project to show'), {
      target: { value: 'packages/alpha' },
    });

    expect(screen.getByRole('heading', { name: 'packages/alpha' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: 'packages/beta' })).not.toBeInTheDocument();
  });

  it('reports the chosen experiment id, not its label', () => {
    const chosen: string[] = [];
    render(
      <ProjectRail
        index={index({ experiments: [experiment({ experiment_id: 'exp:zzz', label: 'train' })] })}
        focusedId=""
        onFocus={(id) => chosen.push(id)}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /train/ }));
    expect(chosen).toEqual(['exp:zzz']);
  });

  it('marks the focused experiment for assistive technology, not just visually', () => {
    render(<ProjectRail index={index()} focusedId="exp:1" onFocus={noop} />);
    expect(screen.getByRole('button', { name: /train/ })).toHaveAttribute('aria-current', 'true');
  });

  it('says where a declared experiment got its name', () => {
    render(
      <ProjectRail
        index={index({ experiments: [experiment({ status: { state: 'active', declared: true } })] })}
        focusedId=""
        onFocus={noop}
      />,
    );
    expect(screen.getByText('declared')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /train/ })).toHaveAttribute(
      'title',
      expect.stringContaining('tensorviz.experiment()'),
    );
  });
});

describe('ProjectRail — a click that has not been answered yet', () => {
  // Clicking sends `projectFocus` and changes nothing on the page: the rail's
  // highlight follows `page.project`, which only moves when the host's document
  // arrives. On a large project that is a long silence, and the only honest
  // reading of a silence is that the click missed — so people click again, and
  // each click queues another document for the host to build.

  it('moves the highlight to the row that was asked for, not the one on screen', () => {
    render(
      <ProjectRail
        index={index({
          experiments: [
            experiment({ experiment_id: 'a', label: 'train' }),
            experiment({ experiment_id: 'b', label: 'evaluate' }),
          ],
        })}
        focusedId="a"
        pendingId="b"
        onFocus={noop}
      />,
    );
    // `aria-current` is what says "this is the one" to everything that is not a
    // pair of eyes, so the assertion is on that rather than on a class name.
    expect(screen.getByRole('button', { name: /evaluate/ })).toHaveAttribute('aria-current', 'true');
    expect(screen.getByRole('button', { name: /train/ })).not.toHaveAttribute('aria-current');
  });

  it('says it is still loading, so chosen is never mistaken for drawn', () => {
    render(
      <ProjectRail
        index={index({ experiments: [experiment({ experiment_id: 'a', label: 'train' })] })}
        focusedId=""
        pendingId="a"
        onFocus={noop}
      />,
    );
    const row = screen.getByRole('button', { name: /train/ });
    expect(row).toHaveAttribute('aria-busy', 'true');
    expect(row).toHaveTextContent('loading…');
    // The status badge is replaced rather than joined: three things competing for
    // one line is what the row has room for, and "active" beside "loading…" reads
    // as a claim about the thing that has not arrived.
    expect(row).not.toHaveTextContent('active');
  });

  it('refuses a second click on the row it is already loading', () => {
    const onFocus = jest.fn();
    render(
      <ProjectRail
        index={index({ experiments: [experiment({ experiment_id: 'a', label: 'train' })] })}
        focusedId=""
        pendingId="a"
        onFocus={onFocus}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: /train/ }));
    expect(onFocus).not.toHaveBeenCalled();
  });

  it('is unchanged when nothing is pending', () => {
    // The counterweight: the resting rail must not gain a spinner or lose a badge.
    render(
      <ProjectRail
        index={index({ experiments: [experiment({ experiment_id: 'a', label: 'train' })] })}
        focusedId="a"
        onFocus={noop}
      />,
    );
    const row = screen.getByRole('button', { name: /train/ });
    expect(row).toHaveAttribute('aria-current', 'true');
    expect(row).not.toHaveAttribute('aria-busy');
    expect(row).toHaveTextContent('active');
  });
});

describe('ProjectRail — finding one experiment among many (step 5.6)', () => {
  // `huggingface/trl` indexes to 138 experiments and really does ship that many
  // recipes, so this is not bad derivation to be fixed elsewhere: it is a list
  // nobody can read without a filter.

  const many = index({
    experiments: [
      experiment({ experiment_id: 'a', label: 'sft', entrypoint: 'trl/scripts/sft.py' }),
      experiment({ experiment_id: 'b', label: 'dpo', entrypoint: 'trl/scripts/dpo.py' }),
      experiment({ experiment_id: 'c', label: 'grpo_math', entrypoint: 'examples/grpo_math.py' }),
      experiment({ experiment_id: 'd', label: 'kto', entrypoint: 'trl/scripts/kto.py' }),
      experiment({ experiment_id: 'e', label: 'reward', entrypoint: 'trl/scripts/reward.py' }),
      experiment({ experiment_id: 'f', label: 'rloo', entrypoint: 'trl/scripts/rloo.py' }),
      experiment({ experiment_id: 'g', label: 'bco', entrypoint: 'trl/scripts/bco.py' }),
      experiment({ experiment_id: 'h', label: 'gkd', entrypoint: 'trl/scripts/gkd.py' }),
      experiment({ experiment_id: 'i', label: 'orpo', entrypoint: 'trl/scripts/orpo.py' }),
    ],
  });

  it('matches on the label, the entrypoint and the root', () => {
    // The three things a person knows about an experiment they are looking for:
    // what it is called, which file runs it, and which project it belongs to.
    expect(experimentMatches(experiment({ label: 'sft' }), 'SF')).toBe(true);
    expect(experimentMatches(experiment({ entrypoint: 'trl/scripts/dpo.py' }), 'scripts')).toBe(true);
    expect(experimentMatches(experiment({ root: 'packages/alpha' }), 'alpha')).toBe(true);
    expect(experimentMatches(experiment({ label: 'sft' }), 'kto')).toBe(false);
    // An empty query is not a filter.
    expect(experimentMatches(experiment({ label: 'sft' }), '   ')).toBe(true);
  });

  it('narrows the list to what was typed', () => {
    render(<ProjectRail index={many} focusedId="" onFocus={noop} />);
    fireEvent.change(screen.getByLabelText('Filter experiments'), { target: { value: 'grpo' } });
    expect(screen.getByRole('button', { name: /grpo_math/ })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /sft/ })).toBeNull();
  });

  it('says how many of how many, so a filtered rail does not look like a lost one', () => {
    render(<ProjectRail index={many} focusedId="" onFocus={noop} />);
    fireEvent.change(screen.getByLabelText('Filter experiments'), { target: { value: 'trl/scripts' } });
    expect(screen.getByText('8 of 9 experiments')).toBeInTheDocument();
  });

  it('says so when nothing matches, rather than showing an empty rail', () => {
    render(<ProjectRail index={many} focusedId="" onFocus={noop} />);
    fireEvent.change(screen.getByLabelText('Filter experiments'), { target: { value: 'nothing' } });
    expect(screen.getByText(/No experiment matches/)).toBeInTheDocument();
  });

  it('offers no filter for a list short enough to read', () => {
    // Below the threshold the list IS the filter, and a search box over six rows
    // is furniture.
    render(<ProjectRail index={index()} focusedId="" onFocus={noop} />);
    expect(screen.queryByLabelText('Filter experiments')).toBeNull();
  });
});
