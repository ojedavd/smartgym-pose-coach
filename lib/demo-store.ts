import type { ReferenceProfile } from '@/lib/pose-analysis';

export interface StoredDemo {
  blob: Blob;
  profile: ReferenceProfile;
  fileName?: string;
}

const DB_NAME = 'forma-motion-coach';
const STORE_NAME = 'exercise-demos';
const DEMO_KEY = 'side-squat';

function openDatabase() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE_NAME)) {
        request.result.createObjectStore(STORE_NAME);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function saveDemo(demo: StoredDemo) {
  const database = await openDatabase();
  await new Promise<void>((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, 'readwrite');
    transaction.objectStore(STORE_NAME).put(demo, DEMO_KEY);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
  });
  database.close();
}

export async function loadDemo() {
  const database = await openDatabase();
  const demo = await new Promise<StoredDemo | undefined>((resolve, reject) => {
    const request = database.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(DEMO_KEY);
    request.onsuccess = () => resolve(request.result as StoredDemo | undefined);
    request.onerror = () => reject(request.error);
  });
  database.close();
  return demo ?? null;
}
