import React from 'react';
import { FlaskConical, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import WorkbenchShell from './WorkbenchShell';

export default function SandboxesPageV2() {
  const navigate = useNavigate();

  const rightSidebar = (
    <div className="space-y-6">
      <section className="ui2-section">
        <div className="ui2-section-head">
          <h3 className="ui2-section-title">Sandbox Actions</h3>
        </div>
        <div>
          <button className="ui2-row" onClick={() => navigate('/builder')}>
            <FlaskConical size={14} />
            Open Sandbox Builder
            <ArrowRight size={14} className="ml-auto ui2-row-glyph" />
          </button>
        </div>
      </section>
    </div>
  );

  return (
    <WorkbenchShell rightSidebar={rightSidebar} rightSidebarTitle="Sandboxes">
      <div className="ui2-page">
        <section className="ui2-section">
          <h1 className="ui2-title mb-6">Sandboxes</h1>
          {/* The two-tone device a second time, one step down the ladder, so it
              reads as a system rather than a hero one-off. The rendered string is
              byte-identical to the paragraph it replaces. */}
          <p className="ui2-lede mb-14">
            <span className="ui2-tone-strong">Explore a new graph</span>{' '}
            and iterate on architecture variants using the layer palette.
          </p>
        </section>
      </div>
    </WorkbenchShell>
  );
}
