// Supabase Apple 제공자의 Secret Key(JWT)를 만든다. Apple 이 준 .p8 키로 ES256 서명, 최대 6개월(Apple 제한) 유효.
// 값은 화면에 찍지 않고 클립보드에만 넣는다(pbcopy). .p8 파일은 레포 밖에 두고 커밋하지 않는다.
// 사용: node scripts/apple_secret.mjs <AuthKey_XXXXXXXXXX.p8 경로> <Key ID> [Team ID] [Services ID]
import { readFileSync } from 'node:fs';
import { createPrivateKey, sign } from 'node:crypto';
import { execFileSync } from 'node:child_process';

const [p8, keyId, teamId = '825X73FW33', clientId = 'me.umjo.ccw.web'] = process.argv.slice(2);
if (!p8 || !keyId) { console.error('사용: node scripts/apple_secret.mjs <.p8 경로> <Key ID> [Team ID] [Services ID]'); process.exit(1); }

const b64u = b => Buffer.from(b).toString('base64url');
const now = Math.floor(Date.now() / 1000), exp = now + 180 * 24 * 3600;   // 180일(6개월 한도 안)
const head = b64u(JSON.stringify({ alg: 'ES256', kid: keyId, typ: 'JWT' }));
const body = b64u(JSON.stringify({ iss: teamId, iat: now, exp, aud: 'https://appleid.apple.com', sub: clientId }));
const sig = sign('sha256', Buffer.from(`${head}.${body}`), { key: createPrivateKey(readFileSync(p8)), dsaEncoding: 'ieee-p1363' });
execFileSync('pbcopy', { input: `${head}.${body}.${b64u(sig)}` });
console.log(`클립보드에 복사함. Supabase → Apple → Secret Key 에 붙여 넣으세요.\n만료: ${new Date(exp * 1000).toLocaleDateString('ko-KR')} (그 전에 다시 만들어 넣어야 함)`);
