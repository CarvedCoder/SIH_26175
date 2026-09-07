/** @module api/results */
import { apiFetch } from './client.js';

/** GET /scenes/{id} */
export const getScene = (id) => apiFetch(`/scenes/${id}`);

/** GET /scenes/{id}/results */
export const getResults = (id) => apiFetch(`/scenes/${id}/results`);

/** GET /scenes/{id}/depth */
export const getDepth = (id) => apiFetch(`/scenes/${id}/depth`);

/** GET /scenes/{id}/dsm */
export const getDsm = (id) => apiFetch(`/scenes/${id}/dsm`);

/** GET /scenes/{id}/reference */
export const getReference = (id) => apiFetch(`/scenes/${id}/reference`);

/** GET /scenes (optional — recent scenes list) */
export const listScenes = () => apiFetch('/scenes');

/** DELETE /scenes/{id} */
export const deleteScene = (id) => apiFetch(`/scenes/${id}`, { method: 'DELETE' });
