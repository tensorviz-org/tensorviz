import { argumentStructure, captureJson, correspondCaptures } from '../../ide/captureCorrespondence.generated';
import fixture from '../../../../tests/fixtures/contracts/portable-capture.json';
import { requirePortableCapture } from '../../services/portableCapture';

it('uses the shared browser-safe correspondence over validated canonical files', () => {
  const before=requirePortableCapture(fixture).capture;
  const after=JSON.parse(JSON.stringify(before));after.id='different-capture';
  const result=correspondCaptures(before,after);
  expect(result.pairs.filter(pair=>pair.kind==='operation').map(pair=>pair.basis)).toEqual(['dataflow_role','dataflow_role']);
  expect(result.unmatched.before.operation).toEqual([]);
});
it('does not treat a duplicate arithmetic operation as the same occurrence', () => {
  const before=requirePortableCapture(fixture).capture;
  const after=JSON.parse(JSON.stringify(before));
  after.operations[1].arguments=JSON.parse(JSON.stringify(after.operations[0].arguments));
  const other=requirePortableCapture({...fixture,capture:after}).capture;
  expect(correspondCaptures(other,other).pairs.filter(pair=>pair.kind==='operation')).toEqual([]);
});
it('preserves kwargs order and scalar types in the common semantic representation', () => {
  const kwargs={a:{kind:'int' as const,value:'1'},b:{kind:'bool' as const,value:true}};
  expect(captureJson(argumentStructure({args:[],kwargs},id=>id))).not.toBe(
    captureJson(argumentStructure({args:[],kwargs:{b:kwargs.b,a:kwargs.a}},id=>id)));
});

it('shares exact field comparison without conflating shapes or weight changes with computation', () => {
  const { compareCaptures } = require('../../ide/captureComparison.generated');
  const before=requirePortableCapture(fixture).capture;
  const after=JSON.parse(JSON.stringify(before));after.id='another-capture';after.stateDigest='b'.repeat(64);
  after.values[0].shape=[4,2];
  const result=compareCaptures(before,after);
  expect(result.equivalent).toBe(true);expect(result.changes).toEqual([]);
  expect(result.observations[0].fields[0].field).toBe('shape');
  expect(result.controls.find((item:any)=>item.field==='weights').status).toBe('different');
  after.operations[0].arguments.args[1].value='2';
  expect(compareCaptures(before,after).changes[0]).toMatchObject({kind:'operation',beforeId:'add1',fields:[{field:'arguments'}]});
});

it('keeps both uses in a+a and identifies the exact reconnected operand', () => {
  const { compareCaptures, capturedValueUses } = require('../../ide/captureComparison.generated');
  const before=JSON.parse(JSON.stringify(fixture.capture));
  before.sourceBindings=[{id:'binding',snapshotId:before.source.id,kind:'call',quality:'exact',anchors:[{
    file:before.source.modelFile,digest:before.source.files[0].digest,startLine:1,startColumn:0,endLine:1,endColumn:5,
    scope:'Model.forward',tokenDigest:'b'.repeat(64)}]}];
  before.operations[1].sourceBindingIds=['binding'];
  const after=JSON.parse(JSON.stringify(before));after.operations[1].arguments.args[1]={kind:'value',id:'h'};
  const a=requirePortableCapture({...fixture,capture:before}).capture,b=requirePortableCapture({...fixture,capture:after}).capture;
  expect(capturedValueUses(a).filter((use:any)=>use.entity.id==='add2').map((use:any)=>use.path)).toEqual([
    ['arguments','args',0],['arguments','args',1],
  ]);
  const result=compareCaptures(a,b);
  expect(result.valueUses.changes).toHaveLength(1);
  expect(result.valueUses.changes[0]).toMatchObject({before:{valueId:'a',path:['arguments','args',1]},after:{valueId:'h'}});
});
