import { PaperTemplateInventoryResponse } from '../../services/apiService';
import { selectPaletteTemplateItems } from '../../components/ui2/BuilderPaletteV2';

function makeInventory(total: number): PaperTemplateInventoryResponse {
  return {
    summary: {
      template_total: total,
      template_builtin: 0,
      template_dynamic: total,
      mapping_total: 0,
      mapping_dynamic: 0,
      staging_pending: 0,
      staging_approved: 0,
      staging_rejected: 0,
    },
    templates: Array.from({ length: total }, (_, index) => ({
      template_id: index === total - 1 ? 'late_audio_tts_block' : `template_${index}`,
      name: index === total - 1 ? 'Late Audio TTS Block' : `Template ${index}`,
      source: 'dynamic',
      required_nodes: 1,
      required_edges: 0,
      training_minimums: 0,
      input_expectation: 'rank-3 tensor [batch, sequence, features]',
    })),
    dynamic_template_ids: [],
    dynamic_mappings: {},
  };
}

describe('selectPaletteTemplateItems', () => {
  it('searches the full template inventory before applying the visible result limit', () => {
    const items = selectPaletteTemplateItems(makeInventory(306), 'late_audio');

    expect(items).toHaveLength(1);
    expect(items[0].template_id).toBe('late_audio_tts_block');
  });

  it('still limits unfiltered results for palette readability', () => {
    const items = selectPaletteTemplateItems(makeInventory(306), '');

    expect(items).toHaveLength(28);
  });
});
